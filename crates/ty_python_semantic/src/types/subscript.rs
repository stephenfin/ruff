use itertools::Itertools;
use ruff_python_ast as ast;

use crate::place::{DefinedPlace, Definedness, Place};
use crate::semantic_index::definition::Definition;
use crate::semantic_index::scope::ScopeId;
use crate::semantic_index::SemanticIndex;
use crate::subscript::{PyIndex, PySlice};
use crate::{Db, FxOrderSet};

use super::call::{Bindings, CallArguments, CallDunderError, CallError, CallErrorKind};
use super::class::KnownClass;
use super::class_base::ClassBase;
use super::context::InferContext;
use super::diagnostic::{
    CALL_NON_CALLABLE, INVALID_ARGUMENT_TYPE, NOT_SUBSCRIPTABLE, POSSIBLY_MISSING_IMPLICIT_CALL,
    report_index_out_of_bounds, report_invalid_key_on_typed_dict, report_not_subscriptable,
    report_slice_step_size_zero,
};
use super::generics::{GenericContext, bind_typevar};
use super::infer::TypeContext;
use super::instance::SliceLiteral;
use super::special_form::SpecialFormType;
use super::tuple::{Tuple, TupleSpec};
use super::visitor::any_over_type;
use super::{DynamicType, KnownInstanceType, Type, TypeAliasType, UnionBuilder, UnionType, todo_type};

#[derive(Debug)]
pub(crate) struct SubscriptError<'db> {
    result_ty: Type<'db>,
    errors: Vec<SubscriptErrorKind<'db>>,
}

#[derive(Debug)]
pub(crate) enum SubscriptErrorKind<'db> {
    /// An index is out of bounds for a literal tuple/string/bytes subscript.
    IndexOutOfBounds {
        kind: &'static str,
        tuple_ty: Type<'db>,
        length: String,
        index: i64,
    },
    /// A slice literal used a step size of zero.
    SliceStepSizeZero,
    /// A non-generic PEP 695 type alias was subscripted.
    NonGenericTypeAlias { alias: TypeAliasType<'db> },
    /// `__getitem__` exists but is possibly unbound.
    DunderPossiblyUnbound {
        method: &'static str,
        value_ty: Type<'db>,
    },
    /// `__getitem__` exists but can't be called with the given arguments.
    DunderCallError {
        method: &'static str,
        value_ty: Type<'db>,
        slice_ty: Type<'db>,
        kind: CallErrorKind,
        bindings: Box<Bindings<'db>>,
    },
    /// `__class_getitem__` exists but isn't callable.
    CallNonCallable {
        method: &'static str,
        value_ty: Type<'db>,
        bindings: Box<Bindings<'db>>,
    },
    /// `__class_getitem__` exists but may be missing at runtime.
    PossiblyMissingImplicitCall {
        method: &'static str,
        value_ty: Type<'db>,
    },
    /// The type does not support subscripting via the expected dunder.
    NotSubscriptable {
        value_ty: Type<'db>,
        method: &'static str,
    },
    /// An invalid argument was provided to `Generic` or `Protocol`.
    InvalidLegacyGenericArgument {
        origin: &'static str,
        argument_ty: Type<'db>,
    },
}

impl<'db> SubscriptError<'db> {
    pub(super) fn new(result_ty: Type<'db>, error: SubscriptErrorKind<'db>) -> Self {
        Self {
            result_ty,
            errors: vec![error],
        }
    }

    pub(super) fn with_errors(result_ty: Type<'db>, errors: Vec<SubscriptErrorKind<'db>>) -> Self {
        Self { result_ty, errors }
    }

    pub(crate) fn result_type(&self) -> Type<'db> {
        self.result_ty
    }

    fn into_errors(self) -> Vec<SubscriptErrorKind<'db>> {
        self.errors
    }

    pub(crate) fn report_diagnostics(
        &self,
        context: &InferContext<'db, '_>,
        subscript: &ast::ExprSubscript,
    ) {
        let value_node = subscript.value.as_ref();
        let slice_node = subscript.slice.as_ref();
        for error in &self.errors {
            error.report_diagnostic(context, subscript, value_node, slice_node);
        }
    }
}

impl<'db> SubscriptErrorKind<'db> {
    fn report_diagnostic(
        &self,
        context: &InferContext<'db, '_>,
        subscript: &ast::ExprSubscript,
        value_node: &ast::Expr,
        slice_node: &ast::Expr,
    ) {
        let db = context.db();
        match self {
            Self::IndexOutOfBounds {
                kind,
                tuple_ty,
                length,
                index,
            } => {
                report_index_out_of_bounds(
                    context,
                    kind,
                    value_node.into(),
                    *tuple_ty,
                    length,
                    *index,
                );
            }
            Self::SliceStepSizeZero => {
                report_slice_step_size_zero(context, value_node.into());
            }
            Self::NonGenericTypeAlias { alias } => {
                if let Some(builder) = context.report_lint(&NOT_SUBSCRIPTABLE, subscript) {
                    let value_type = alias.raw_value_type(db);
                    let mut diagnostic =
                        builder.into_diagnostic("Cannot subscript non-generic type alias");
                    if value_type.is_definition_generic(db) {
                        diagnostic.set_primary_message(format_args!(
                            "`{}` is already specialized",
                            value_type.display(db)
                        ));
                    }
                }
            }
            Self::DunderPossiblyUnbound { method, value_ty } => {
                if let Some(builder) =
                    context.report_lint(&POSSIBLY_MISSING_IMPLICIT_CALL, value_node)
                {
                    builder.into_diagnostic(format_args!(
                        "Method `{method}` of type `{}` may be missing",
                        value_ty.display(db),
                    ));
                }
            }
            Self::DunderCallError {
                method,
                value_ty,
                slice_ty,
                kind,
                bindings,
            } => match kind {
                CallErrorKind::NotCallable => {
                    if let Some(builder) = context.report_lint(&CALL_NON_CALLABLE, value_node) {
                        builder.into_diagnostic(format_args!(
                            "Method `{method}` of type `{}` is not callable on object of type `{}`",
                            bindings.callable_type().display(db),
                            value_ty.display(db),
                        ));
                    }
                }
                CallErrorKind::BindingError => {
                    if let Some(typed_dict) = value_ty.as_typed_dict() {
                        report_invalid_key_on_typed_dict(
                            context,
                            value_node.into(),
                            slice_node.into(),
                            *value_ty,
                            None,
                            *slice_ty,
                            typed_dict.items(db),
                        );
                    } else if let Some(builder) =
                        context.report_lint(&INVALID_ARGUMENT_TYPE, value_node)
                    {
                        builder.into_diagnostic(format_args!(
                            "Method `{method}` of type `{}` cannot be called with key of type `{}` on object of type `{}`",
                            bindings.callable_type().display(db),
                            slice_ty.display(db),
                            value_ty.display(db),
                        ));
                    }
                }
                CallErrorKind::PossiblyNotCallable => {
                    if let Some(builder) = context.report_lint(&CALL_NON_CALLABLE, value_node) {
                        builder.into_diagnostic(format_args!(
                            "Method `{method}` of type `{}` may not be callable on object of type `{}`",
                            bindings.callable_type().display(db),
                            value_ty.display(db),
                        ));
                    }
                }
            },
            Self::CallNonCallable {
                method,
                value_ty,
                bindings,
            } => {
                if let Some(builder) = context.report_lint(&CALL_NON_CALLABLE, value_node) {
                    builder.into_diagnostic(format_args!(
                        "Method `{method}` of type `{}` is not callable on object of type `{}`",
                        bindings.callable_type().display(db),
                        value_ty.display(db),
                    ));
                }
            }
            Self::PossiblyMissingImplicitCall { method, value_ty } => {
                if let Some(builder) =
                    context.report_lint(&POSSIBLY_MISSING_IMPLICIT_CALL, value_node)
                {
                    builder.into_diagnostic(format_args!(
                        "Method `{method}` of type `{}` may be missing",
                        value_ty.display(db),
                    ));
                }
            }
            Self::NotSubscriptable { value_ty, method } => {
                report_not_subscriptable(context, subscript, *value_ty, method);
            }
            Self::InvalidLegacyGenericArgument {
                origin,
                argument_ty,
            } => {
                if let Some(builder) = context.report_lint(&INVALID_ARGUMENT_TYPE, value_node) {
                    builder.into_diagnostic(format_args!(
                        "`{}` is not a valid argument to `{origin}`",
                        argument_ty.display(db),
                    ));
                }
            }
        }
    }
}

pub(super) fn map_union_subscript<'db, F>(
    db: &'db dyn Db,
    union: UnionType<'db>,
    mut map_fn: F,
) -> Result<Type<'db>, SubscriptError<'db>>
where
    F: FnMut(Type<'db>) -> Result<Type<'db>, SubscriptError<'db>>,
{
    let mut builder = UnionBuilder::new(db);
    let mut errors = Vec::new();

    for element in union.elements(db) {
        match map_fn(*element) {
            Ok(result) => {
                builder = builder.add(result);
            }
            Err(error) => {
                builder = builder.add(error.result_type());
                errors.extend(error.into_errors());
            }
        }
    }

    builder = builder.recursively_defined(union.recursively_defined(db));
    let result_ty = builder.build();
    if errors.is_empty() {
        Ok(result_ty)
    } else {
        Err(SubscriptError::with_errors(result_ty, errors))
    }
}

impl<'db> Type<'db> {
    pub(super) fn subscript(
        self,
        db: &'db dyn Db,
        slice_ty: Type<'db>,
        expr_context: ast::ExprContext,
        scope_id: ScopeId<'db>,
        index: &'db SemanticIndex<'db>,
        typevar_binding_context: Option<Definition<'db>>,
    ) -> Result<Type<'db>, SubscriptError<'db>> {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum LegacyGenericContextError<'db> {
            /// It's invalid to subscript `Generic` or `Protocol` with this type.
            InvalidArgument(Type<'db>),
            /// It's invalid to subscript `Generic` or `Protocol` with a variadic tuple type.
            /// We should emit a diagnostic for this, but we don't yet.
            VariadicTupleArguments,
            /// It's valid to subscribe `Generic` or `Protocol` with this type,
            /// but the type is not yet supported.
            NotYetSupported,
        }

        impl<'db> LegacyGenericContextError<'db> {
            const fn into_type(self) -> Type<'db> {
                match self {
                    LegacyGenericContextError::InvalidArgument(_)
                    | LegacyGenericContextError::VariadicTupleArguments => Type::unknown(),
                    LegacyGenericContextError::NotYetSupported => {
                        todo_type!("ParamSpecs and TypeVarTuples")
                    }
                }
            }
        }

        let legacy_generic_class_context = |typevars: Type<'db>| -> Result<
            GenericContext<'db>,
            LegacyGenericContextError<'db>,
        > {
            let typevars_class_tuple_spec = typevars.exact_tuple_instance_spec(db);

            let typevars = if let Some(tuple_spec) = typevars_class_tuple_spec.as_deref() {
                match tuple_spec {
                    Tuple::Fixed(typevars) => typevars.elements_slice(),
                    Tuple::Variable(_) => {
                        return Err(LegacyGenericContextError::VariadicTupleArguments);
                    }
                }
            } else {
                std::slice::from_ref(&typevars)
            };

            let typevars: Result<FxOrderSet<_>, LegacyGenericContextError<'db>> = typevars
                .iter()
                .map(|typevar| {
                    let argument_ty = *typevar;
                    if let Type::KnownInstance(KnownInstanceType::TypeVar(typevar)) = argument_ty {
                        bind_typevar(
                            db,
                            index,
                            scope_id.file_scope_id(db),
                            typevar_binding_context,
                            typevar,
                        )
                        .ok_or(LegacyGenericContextError::InvalidArgument(argument_ty))
                    } else if any_over_type(
                        db,
                        argument_ty,
                        &|ty| match ty {
                            Type::Dynamic(
                                DynamicType::TodoUnpack | DynamicType::TodoStarredExpression,
                            ) => true,
                            Type::NominalInstance(nominal) => {
                                nominal.has_known_class(db, KnownClass::TypeVarTuple)
                            }
                            _ => false,
                        },
                        true,
                    ) {
                        Err(LegacyGenericContextError::NotYetSupported)
                    } else {
                        Err(LegacyGenericContextError::InvalidArgument(argument_ty))
                    }
                })
                .collect();
            typevars.map(|typevars| GenericContext::from_typevar_instances(db, typevars))
        };

        let value_ty = self;

        let inferred = match (value_ty, slice_ty) {
            (Type::Dynamic(_) | Type::Never, _) => Some(Ok(value_ty)),

            (Type::TypeAlias(alias), _) => Some(alias.value_type(db).subscript(
                db,
                slice_ty,
                expr_context,
                scope_id,
                index,
                typevar_binding_context,
            )),

            (_, Type::TypeAlias(alias)) => Some(value_ty.subscript(
                db,
                alias.value_type(db),
                expr_context,
                scope_id,
                index,
                typevar_binding_context,
            )),

            (Type::Union(union), _) => Some(map_union_subscript(db, union, |element| {
                element.subscript(
                    db,
                    slice_ty,
                    expr_context,
                    scope_id,
                    index,
                    typevar_binding_context,
                )
            })),

            (_, Type::Union(union)) => Some(map_union_subscript(db, union, |element| {
                value_ty.subscript(
                    db,
                    element,
                    expr_context,
                    scope_id,
                    index,
                    typevar_binding_context,
                )
            })),

            (Type::Intersection(_), _) | (_, Type::Intersection(_)) => {
                Some(Ok(todo_type!("Subscript expressions with intersections")))
            }

            // Ex) Given `("a", "b", "c", "d")[1]`, return `"b"`
            (Type::NominalInstance(nominal), Type::IntLiteral(i64_int)) => {
                nominal
                    .tuple_spec(db)
                    .and_then(|tuple| Some((tuple, i32::try_from(i64_int).ok()?)))
                    .map(|(tuple, i32_int)| match tuple.py_index(db, i32_int) {
                        Ok(result) => Ok(result),
                        Err(_) => Err(SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::IndexOutOfBounds {
                                kind: "tuple",
                                tuple_ty: value_ty,
                                length: tuple.len().display_minimum(),
                                index: i64_int,
                            },
                        )),
                    })
            }

            // Ex) Given `("a", 1, Null)[0:2]`, return `("a", 1)`
            (
                Type::NominalInstance(maybe_tuple_nominal),
                Type::NominalInstance(maybe_slice_nominal),
            ) => maybe_tuple_nominal
                .tuple_spec(db)
                .as_deref()
                .and_then(|tuple_spec| Some((tuple_spec, maybe_slice_nominal.slice_literal(db)?)))
                .map(|(tuple, SliceLiteral { start, stop, step })| match tuple {
                    TupleSpec::Fixed(tuple) => match tuple.py_slice(db, start, stop, step) {
                        Ok(new_elements) => {
                            Ok(Type::heterogeneous_tuple(db, new_elements))
                        }
                        Err(_) => Err(SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::SliceStepSizeZero,
                        )),
                    },
                    TupleSpec::Variable(_) => {
                        Ok(todo_type!("slice into variable-length tuple"))
                    }
                }),

            // Ex) Given `"value"[1]`, return `"a"`
            (Type::StringLiteral(literal_ty), Type::IntLiteral(i64_int)) => {
                i32::try_from(i64_int).ok().map(|i32_int| {
                    let literal_value = literal_ty.value(db);
                    match (&mut literal_value.chars()).py_index(db, i32_int) {
                        Ok(ch) => Ok(Type::string_literal(db, &ch.to_string())),
                        Err(_) => Err(SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::IndexOutOfBounds {
                                kind: "string",
                                tuple_ty: value_ty,
                                length: literal_value.chars().count().to_string(),
                                index: i64_int,
                            },
                        )),
                    }
                })
            }

            // Ex) Given `"value"[1:3]`, return `"al"`
            (Type::StringLiteral(literal_ty), Type::NominalInstance(nominal)) => nominal
                .slice_literal(db)
                .map(|SliceLiteral { start, stop, step }| {
                    let literal_value = literal_ty.value(db);
                    let chars: Vec<_> = literal_value.chars().collect();

                    match chars.py_slice(db, start, stop, step) {
                        Ok(new_chars) => {
                            let literal = new_chars.collect::<String>();
                            Ok(Type::string_literal(db, &literal))
                        }
                        Err(_) => Err(SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::SliceStepSizeZero,
                        )),
                    }
                }),

            (Type::LiteralString, Type::IntLiteral(_) | Type::BooleanLiteral(_)) => {
                Some(Ok(Type::LiteralString))
            }

            (Type::LiteralString, Type::NominalInstance(nominal))
                if nominal.slice_literal(db).is_some() =>
            {
                Some(Ok(Type::LiteralString))
            }

            // Ex) Given `b"value"[1]`, return `97` (i.e., `ord(b"a")`)
            (Type::BytesLiteral(literal_ty), Type::IntLiteral(i64_int)) => {
                i32::try_from(i64_int).ok().map(|i32_int| {
                    let literal_value = literal_ty.value(db);
                    match literal_value.py_index(db, i32_int) {
                        Ok(byte) => Ok(Type::IntLiteral((*byte).into())),
                        Err(_) => Err(SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::IndexOutOfBounds {
                                kind: "bytes literal",
                                tuple_ty: value_ty,
                                length: literal_value.len().to_string(),
                                index: i64_int,
                            },
                        )),
                    }
                })
            }

            // Ex) Given `b"value"[1:3]`, return `b"al"`
            (Type::BytesLiteral(literal_ty), Type::NominalInstance(nominal)) => nominal
                .slice_literal(db)
                .map(|SliceLiteral { start, stop, step }| {
                    let literal_value = literal_ty.value(db);

                    match literal_value.py_slice(db, start, stop, step) {
                        Ok(new_bytes) => {
                            let new_bytes = new_bytes.collect::<Vec<u8>>();
                            Ok(Type::bytes_literal(db, &new_bytes))
                        }
                        Err(_) => Err(SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::SliceStepSizeZero,
                        )),
                    }
                }),

            // Ex) Given `"value"[True]`, return `"a"`
            (Type::StringLiteral(_) | Type::BytesLiteral(_), Type::BooleanLiteral(bool)) => {
                Some(value_ty.subscript(
                    db,
                    Type::IntLiteral(i64::from(bool)),
                    expr_context,
                    scope_id,
                    index,
                    typevar_binding_context,
                ))
            }

            (Type::NominalInstance(nominal), Type::BooleanLiteral(bool))
                if nominal.tuple_spec(db).is_some() =>
            {
                Some(value_ty.subscript(
                    db,
                    Type::IntLiteral(i64::from(bool)),
                    expr_context,
                    scope_id,
                    index,
                    typevar_binding_context,
                ))
            }

            (Type::SpecialForm(SpecialFormType::Protocol), typevars) => Some(
                match legacy_generic_class_context(typevars) {
                    Ok(context) => Ok(Type::KnownInstance(
                        KnownInstanceType::SubscriptedProtocol(context),
                    )),
                    Err(LegacyGenericContextError::InvalidArgument(argument_ty)) => Err(
                        SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::InvalidLegacyGenericArgument {
                                origin: "Protocol",
                                argument_ty,
                            },
                        ),
                    ),
                    Err(error) => Ok(error.into_type()),
                },
            ),

            (Type::KnownInstance(KnownInstanceType::SubscriptedProtocol(_)), _) => {
                // TODO: emit a diagnostic
                Some(Ok(todo_type!("doubly-specialized typing.Protocol")))
            }

            (
                Type::KnownInstance(KnownInstanceType::TypeAliasType(TypeAliasType::PEP695(alias))),
                _,
            ) if alias.generic_context(db).is_none() => {
                debug_assert!(alias.specialization(db).is_none());
                Some(Err(SubscriptError::new(
                    Type::unknown(),
                    SubscriptErrorKind::NonGenericTypeAlias {
                        alias: TypeAliasType::PEP695(alias),
                    },
                )))
            }

            (Type::SpecialForm(SpecialFormType::Generic), typevars) => Some(
                match legacy_generic_class_context(typevars) {
                    Ok(context) => Ok(Type::KnownInstance(
                        KnownInstanceType::SubscriptedGeneric(context),
                    )),
                    Err(LegacyGenericContextError::InvalidArgument(argument_ty)) => Err(
                        SubscriptError::new(
                            Type::unknown(),
                            SubscriptErrorKind::InvalidLegacyGenericArgument {
                                origin: "Generic",
                                argument_ty,
                            },
                        ),
                    ),
                    Err(error) => Ok(error.into_type()),
                },
            ),

            (Type::KnownInstance(KnownInstanceType::SubscriptedGeneric(_)), _) => {
                // TODO: emit a diagnostic
                Some(Ok(todo_type!("doubly-specialized typing.Generic")))
            }

            (Type::SpecialForm(SpecialFormType::Unpack), _) => {
                Some(Ok(Type::Dynamic(DynamicType::TodoUnpack)))
            }

            (Type::SpecialForm(SpecialFormType::Concatenate), _) => {
                // TODO: Add proper support for `Concatenate`
                let mut variables = FxOrderSet::default();
                slice_ty.bind_and_find_all_legacy_typevars(
                    db,
                    typevar_binding_context,
                    &mut variables,
                );
                let generic_context = GenericContext::from_typevar_instances(db, variables);
                Some(Ok(Type::Dynamic(DynamicType::UnknownGeneric(generic_context))))
            }

            (Type::SpecialForm(special_form), _) if special_form.class().is_special_form() => {
                Some(Ok(todo_type!("Inference of subscript on special form")))
            }

            (Type::KnownInstance(known_instance), _) if known_instance.class(db).is_special_form() => {
                Some(Ok(todo_type!("Inference of subscript on special form")))
            }

            (
                Type::FunctionLiteral(_)
                | Type::WrapperDescriptor(_)
                | Type::BoundMethod(_)
                | Type::DataclassDecorator(_)
                | Type::DataclassTransformer(_)
                | Type::Callable(_)
                | Type::ModuleLiteral(_)
                | Type::ClassLiteral(_)
                | Type::GenericAlias(_)
                | Type::SubclassOf(_)
                | Type::AlwaysFalsy
                | Type::AlwaysTruthy
                | Type::IntLiteral(_)
                | Type::BooleanLiteral(_)
                | Type::ProtocolInstance(_)
                | Type::PropertyInstance(_)
                | Type::EnumLiteral(_)
                | Type::BoundSuper(_)
                | Type::TypeIs(_)
                | Type::TypeGuard(_)
                | Type::TypedDict(_)
                | Type::NewTypeInstance(_)
                | Type::NominalInstance(_)
                | Type::SpecialForm(_)
                | Type::KnownInstance(_)
                | Type::StringLiteral(_)
                | Type::BytesLiteral(_)
                | Type::LiteralString
                | Type::TypeVar(_)  // TODO: more complex logic required here!
                | Type::KnownBoundMethod(_),
                _,
            ) => None,
        };

        if let Some(inferred) = inferred {
            return inferred;
        }

        // If the class defines `__getitem__`, return its return type.
        //
        // See: https://docs.python.org/3/reference/datamodel.html#class-getitem-versus-getitem
        match value_ty.try_call_dunder(
            db,
            "__getitem__",
            CallArguments::positional([slice_ty]),
            TypeContext::default(),
        ) {
            Ok(outcome) => {
                return Ok(outcome.return_type(db));
            }
            Err(CallDunderError::PossiblyUnbound(bindings)) => {
                return Err(SubscriptError::new(
                    bindings.return_type(db),
                    SubscriptErrorKind::DunderPossiblyUnbound {
                        method: "__getitem__",
                        value_ty,
                    },
                ));
            }
            Err(CallDunderError::CallError(call_error_kind, bindings)) => {
                return Err(SubscriptError::new(
                    bindings.return_type(db),
                    SubscriptErrorKind::DunderCallError {
                        method: "__getitem__",
                        value_ty,
                        slice_ty,
                        kind: call_error_kind,
                        bindings,
                    },
                ));
            }
            Err(CallDunderError::MethodNotAvailable) => {
                // try `__class_getitem__`
            }
        }

        // Otherwise, if the value is itself a class and defines `__class_getitem__`,
        // return its return type.
        //
        // TODO: lots of classes are only subscriptable at runtime on Python 3.9+,
        // *but* we should also allow them to be subscripted in stubs
        // (and in annotations if `from __future__ import annotations` is enabled),
        // even if the target version is Python 3.8 or lower,
        // despite the fact that there will be no corresponding `__class_getitem__`
        // method in these `sys.version_info` branches.
        if value_ty.is_subtype_of(db, KnownClass::Type.to_instance(db)) {
            let dunder_class_getitem_method = value_ty.member(db, "__class_getitem__").place;

            match dunder_class_getitem_method {
                Place::Undefined => {}
                Place::Defined(DefinedPlace {
                    ty,
                    definedness: boundness,
                    ..
                }) => {
                    let mut errors = Vec::new();
                    if boundness == Definedness::PossiblyUndefined {
                        errors.push(SubscriptErrorKind::PossiblyMissingImplicitCall {
                            method: "__class_getitem__",
                            value_ty,
                        });
                    }

                    match ty.try_call(db, &CallArguments::positional([slice_ty])) {
                        Ok(bindings) => {
                            let result_ty = bindings.return_type(db);
                            if errors.is_empty() {
                                return Ok(result_ty);
                            }
                            return Err(SubscriptError::with_errors(result_ty, errors));
                        }
                        Err(CallError(_, bindings)) => {
                            let result_ty = bindings.return_type(db);
                            errors.push(SubscriptErrorKind::CallNonCallable {
                                method: "__class_getitem__",
                                value_ty,
                                bindings,
                            });
                            return Err(SubscriptError::with_errors(result_ty, errors));
                        }
                    }
                }
            }

            if let Type::ClassLiteral(class) = value_ty {
                if class.is_known(db, KnownClass::Type) {
                    return Ok(KnownClass::GenericAlias.to_instance(db));
                }

                if class.generic_context(db).is_some() {
                    // TODO: specialize the generic class using these explicit type
                    // variable assignments. This branch is only encountered when an
                    // explicit class specialization appears inside of some other subscript
                    // expression, e.g. `tuple[list[int], ...]`. We have already inferred
                    // the type of the outer subscript slice as a value expression, which
                    // means we can't re-infer the inner specialization here as a type
                    // expression.
                    return Ok(value_ty);
                }
            }

            // TODO: properly handle old-style generics; get rid of this temporary hack
            if !value_ty
                .as_class_literal()
                .is_some_and(|class| class.iter_mro(db).contains(&ClassBase::Generic))
            {
                return Err(SubscriptError::new(
                    Type::unknown(),
                    SubscriptErrorKind::NotSubscriptable {
                        value_ty,
                        method: "__class_getitem__",
                    },
                ));
            }
        } else if expr_context != ast::ExprContext::Store {
            return Err(SubscriptError::new(
                Type::unknown(),
                SubscriptErrorKind::NotSubscriptable {
                    value_ty,
                    method: "__getitem__",
                },
            ));
        }

        Ok(Type::unknown())
    }
}

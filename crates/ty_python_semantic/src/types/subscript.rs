use ruff_python_ast as ast;

use crate::Db;

use super::call::{Bindings, CallErrorKind};
use super::context::InferContext;
use super::diagnostic::{
    CALL_NON_CALLABLE, INVALID_ARGUMENT_TYPE, NOT_SUBSCRIPTABLE, POSSIBLY_MISSING_IMPLICIT_CALL,
    report_index_out_of_bounds, report_invalid_key_on_typed_dict, report_not_subscriptable,
    report_slice_step_size_zero,
};
use super::{Type, TypeAliasType, UnionBuilder, UnionType};

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

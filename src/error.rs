//! Error type for graph manipulation.
use std::fmt;

/// Error raised by [`Graph`](super::Graph).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphError {
    /// The handle is marked as dead (the node or arc was deleted)
    DeadHandle,
    /// The handle does not belong to this graph
    WrongGraph,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for GraphError {}

/// Result of [`Graph`](super::Graph) operations.
pub type Result<T> = std::result::Result<T, GraphError>;

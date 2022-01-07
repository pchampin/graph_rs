//! A change list stores a list of mutations,
//! that will later be applied to a [`Graph`](crate::Graph).
//! This is useful because mutating a graph while browsing it would be unsafe.

use super::*;

/// A list of changes to be [applied][`Graph::apply`] to a [`Graph`].
pub struct ChangeList<N, A> {
    changes: Vec<Change<N, A>>,
    new_nodes: usize,
    new_arcs: usize,
}

enum Change<N, A> {
    NewNode(N),
    NewArc(PrFt<NodeHandle<N, A>>, PrFt<NodeHandle<N, A>>, A),
    DeleteNode(PrFt<NodeHandle<N, A>>),
    DeleteArc(PrFt<ArcHandle<N, A>>),
    MutateNode(PrFt<NodeHandle<N, A>>, Box<dyn FnOnce(&mut N) + 'static>),
    MutateArc(PrFt<ArcHandle<N, A>>, Box<dyn FnOnce(&mut A) + 'static>),
}

/// A [`Handle`] that either exist in the present, or will exist in the future.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PrFt<T> {
    /// A [`Handle`] that already exists.
    Present(T),
    /// A [`Handle`]-like object that will exist as a [`Handle`] in the future.
    Future(usize),
}

use Change::*;
use PrFt::*;

//

impl<N, A> ChangeList<N, A> {
    /// Create a new empty change list.
    pub fn new() -> Self {
        ChangeList::default()
    }

    /// Create a new node carrying the given data.
    pub fn new_node_with(&mut self, data: N) -> PrFt<NodeHandle<N, A>> {
        let i = self.new_nodes;
        self.new_nodes += 1;
        self.changes.push(NewNode(data));
        Future(i)
    }

    /// Create a new node carrying the default data.
    pub fn new_node(&mut self) -> PrFt<NodeHandle<N, A>>
    where
        N: Default,
    {
        self.new_node_with(N::default())
    }

    /// Create a new arc from `src` to `dst`, carrying the given data.
    pub fn new_arc_with<T, U>(&mut self, src: T, dst: U, data: A) -> PrFt<ArcHandle<N, A>>
    where
        T: Into<PrFt<NodeHandle<N, A>>>,
        U: Into<PrFt<NodeHandle<N, A>>>,
    {
        let i = self.new_arcs;
        self.new_arcs += 1;
        self.changes.push(NewArc(src.into(), dst.into(), data));
        Future(i)
    }

    /// Create a new arc from `src` to `dst`, carrying the default data.
    pub fn new_arc<T, U>(&mut self, src: T, dst: U) -> PrFt<ArcHandle<N, A>>
    where
        A: Default,
        T: Into<PrFt<NodeHandle<N, A>>>,
        U: Into<PrFt<NodeHandle<N, A>>>,
    {
        self.new_arc_with(src, dst, A::default())
    }

    /// Delete the corresponding node.
    pub fn delete_node<T>(&mut self, node: T)
    where
        T: Into<PrFt<NodeHandle<N, A>>>,
    {
        self.changes.push(DeleteNode(node.into()))
    }

    /// Delete the corresponding arc.
    pub fn delete_arc<T>(&mut self, arc: T)
    where
        T: Into<PrFt<ArcHandle<N, A>>>,
    {
        self.changes.push(DeleteArc(arc.into()))
    }

    /// Apply the mutating function to the corresponding node's data.
    pub fn mutate_node<T, F>(&mut self, node: T, f: F)
    where
        T: Into<PrFt<NodeHandle<N, A>>>,
        F: FnOnce(&mut N) + 'static,
    {
        self.changes.push(MutateNode(node.into(), Box::new(f)))
    }

    /// Apply the mutating function to the corresponding arc's data.
    pub fn mutate_arc<T, F>(&mut self, arc: T, f: F)
    where
        T: Into<PrFt<ArcHandle<N, A>>>,
        F: FnOnce(&mut A) + 'static,
    {
        self.changes.push(MutateArc(arc.into(), Box::new(f)))
    }

    pub(super) fn apply_to(self, g: &mut Graph<N, A>) {
        let mut new_nodes = Vec::with_capacity(self.new_nodes);
        let mut new_arcs = Vec::with_capacity(self.new_arcs);
        for change in self.changes {
            match change {
                NewNode(data) => {
                    new_nodes.push(g.new_node_with(data));
                }
                NewArc(src, dst, data) => {
                    new_arcs.push(g.new_arc_with(
                        src.handle(&new_nodes),
                        dst.handle(&new_nodes),
                        data,
                    ));
                }
                DeleteNode(prft) => {
                    g.delete_node(prft.handle(&new_nodes));
                }
                DeleteArc(prft) => {
                    if let Some(handle) = prft.try_handle(&new_arcs) {
                        g.delete_arc(handle)
                    }
                }
                MutateNode(prft, f) => {
                    if let Some(node) = g.node_mut(prft.handle(&new_nodes)) {
                        f(node.data_mut())
                    }
                }
                MutateArc(prft, f) => {
                    if let Some(arc) = prft
                        .try_handle(&new_arcs)
                        .and_then(|handle| g.arc_mut(handle))
                    {
                        f(arc.data_mut())
                    }
                }
            }
        }
    }
}

impl<N, A> Default for ChangeList<N, A> {
    fn default() -> Self {
        ChangeList {
            changes: vec![],
            new_nodes: 0,
            new_arcs: 0,
        }
    }
}

//

impl<T> PrFt<T> {
    fn handle<'a>(&'a self, created: &'a [T]) -> &'a T {
        match self {
            Present(handle) => handle,
            Future(i) => &created[*i],
        }
    }

    fn try_handle<'a>(&'a self, created: &'a [Option<T>]) -> Option<&'a T> {
        match self {
            Present(handle) => Some(handle),
            Future(i) => created[*i].as_ref(),
        }
    }
}

impl<'a, T> From<&'a T> for PrFt<T>
where
    T: Clone,
{
    fn from(other: &'a T) -> PrFt<T> {
        Present(other.clone())
    }
}

impl<'a, T> From<&'a PrFt<T>> for PrFt<T>
where
    T: Clone,
{
    fn from(other: &'a PrFt<T>) -> PrFt<T> {
        other.clone()
    }
}

impl<T> PartialEq<T> for PrFt<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &T) -> bool {
        match self {
            Present(nt) => nt == other,
            Future(_) => false,
        }
    }
}

impl<N, A> PartialEq<PrFt<NodeHandle<N, A>>> for NodeHandle<N, A> {
    fn eq(&self, other: &PrFt<NodeHandle<N, A>>) -> bool {
        other == self
    }
}

impl<N, A> PartialEq<PrFt<ArcHandle<N, A>>> for ArcHandle<N, A> {
    fn eq(&self, other: &PrFt<ArcHandle<N, A>>) -> bool {
        other == self
    }
}

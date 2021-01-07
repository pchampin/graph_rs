//! A lock-free [`Graph`] library for Rust
//!
//! Graphs, nodes and arcs are always boxed,
//! so their memory address is stable along their whole life.
//! (They can not be used as identifiers, though,
//! because they can be *reused* after deletion.)
//!
//! The graph owns its nodes, which in turn own their outgoing arcs.
//!
//! Mutable references to Nodes and Arcs allow to change their internal data,
//! and to get mutable references to neighbouring Nodes and Arcs,
//! but they *do not* allow to change the structure of the graph
//! (as this would impact other nodes or arcs).
//!
//! To change the structure of the graph,
//! one must have a mutable reference to the graph itself,
//! and *handles* to the involved nodes and/or arcs.
//!
//! Handles can be obtained from Nodes and Arcs themselves,
//! even through immutable references.
//! They contain a pointer to the corresponding node or arcs,
//! so they provide a almost-zero-cost abstraction.
//! *Almost* zero, because two checks must still be performed:
//! * whether the pointed element belongs to the graph to which they are passed, and
//! * whether the pointed elements has been deleted.
//! Both checks are done in constant time, though.
//!
//! Finally, the structure of a graph can not be modified while browsing it.
//! Structural changes can however be stored in a [`ChangeList`],
//! which can later be [applied](Graph::apply) to the graph.
#![deny(missing_docs)]

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::sync;
use std::sync::atomic::{AtomicBool, Ordering};

pub mod changelist;
pub use changelist::ChangeList;

/// A graph whose nodes contain `N` data, and whose arcs contain `A` data.
#[derive(Clone, Debug)]
pub struct Graph<N, A> {
    nodes: Vec<Box<Node<N, A>>>,
}

/// A node belonging to a [`Graph`].
#[derive(Clone, Debug)]
pub struct Node<N, A> {
    graph: NonNull<Graph<N, A>>,
    data: N,
    out_arcs: Vec<Box<Arc<N, A>>>,
    in_arcs: Vec<NonNull<Arc<N, A>>>,
    handle: NodeHandle<N, A>,
}

/// An arc belonging to a [`Graph`].
#[derive(Clone, Debug)]
pub struct Arc<N, A> {
    src: NonNull<Node<N, A>>,
    dst: NonNull<Node<N, A>>,
    data: A,
    handle: ArcHandle<N, A>,
}

/// A handle to a [`Graph`] element ([`Node`] or [`Arc`]).
///
/// The handle is basically a pointer with a security,
/// allowing to verify in constant time that the pointed element is still alive.
///
/// See [module documentation](crate) for more details.
pub struct Handle<T> {
    ptr: NonNull<T>,
    alive: AtomicBool,
}

/// Type alias for [`Node`] [`Handle`].
pub type NodeHandle<N, A> = sync::Arc<Handle<Node<N, A>>>;

/// Type alias for [`Arc`] [`Handle`].
pub type ArcHandle<N, A> = sync::Arc<Handle<Arc<N, A>>>;

//

impl<N, A> Graph<N, A> {
    /// Create a new empty graph.
    pub fn new() -> Box<Graph<N, A>> {
        Box::new(Graph { nodes: vec![] })
    }

    /// The number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether this graph contains no node.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over the nodes of this graph.
    pub fn nodes(&self) -> impl Iterator<Item = &Node<N, A>> {
        self.nodes.iter().map(Box::as_ref)
    }

    /// Iterate over the nodes (as mutable references) of this graph.
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node<N, A>> {
        self.nodes.iter_mut().map(Box::as_mut)
    }

    /// Return a reference to the node corresponding to this handle.
    ///
    /// This method may fail (return `None`) if:
    /// * the handle is not from this graph, or
    /// * the corresponding node has been deleted from the graph since the handle was issued.
    pub fn node(&self, handle: &NodeHandle<N, A>) -> Option<&Node<N, A>> {
        if self.check_node_handle(handle) {
            Some(unsafe { &*handle.ptr.as_ptr() })
        } else {
            None
        }
    }

    /// Return a mutable reference to the node corresponding to this handle.
    ///
    /// See [`node`](Graph::node) for situations where this method may fail (return `None`).
    pub fn node_mut(&mut self, handle: &NodeHandle<N, A>) -> Option<&mut Node<N, A>> {
        if self.check_node_handle(handle) {
            Some(unsafe { &mut *handle.ptr.as_ptr() })
        } else {
            None
        }
    }

    /// Return a reference to the arc corresponding to this handle.
    ///
    /// See [`node`](Graph::node) for situations where this method may fail (return `None`).
    pub fn arc(&self, handle: &ArcHandle<N, A>) -> Option<&Arc<N, A>> {
        if self.check_arc_handle(handle) {
            Some(unsafe { &*handle.ptr.as_ptr() })
        } else {
            None
        }
    }

    /// Return a mutable reference to the arc corresponding to this handle.
    ///
    /// See [`node`](Graph::node) for situations where this method may fail (return `None`).
    pub fn arc_mut(&mut self, handle: &ArcHandle<N, A>) -> Option<&mut Arc<N, A>> {
        if self.check_arc_handle(handle) {
            Some(unsafe { &mut *handle.ptr.as_ptr() })
        } else {
            None
        }
    }

    /// Create a new node in this graph, carrying the given data.
    ///
    /// Return a [`Handle`] to the newly created node.
    pub fn new_node_with(&mut self, data: N) -> NodeHandle<N, A> {
        let mut new = Box::new(Node {
            graph: NonNull::from(self as &Self),
            data,
            out_arcs: vec![],
            in_arcs: vec![],
            handle: Handle::dummy(),
        });
        new.handle = Handle::new(&new);
        let handle = new.handle();
        self.nodes.push(new);
        handle
    }

    /// Create a new node in this graph, carrying the default data.
    ///
    /// Return a [`Handle`] to the newly created node.
    pub fn new_node(&mut self) -> NodeHandle<N, A>
    where
        N: Default,
    {
        self.new_node_with(N::default())
    }

    /// Add a new arc from `src` to `dst`, carrying the given data.
    ///
    /// Return a [`Handle`] to the newly created arc.
    ///
    /// # Failure to create an arc
    /// The given node handles may fail to identify nodes of the graph
    /// (see [`node`](Graph::node) for more details)
    /// in which case the arc creation will fail, and `None` will be returned.
    pub fn new_arc_with(
        &mut self,
        src: &NodeHandle<N, A>,
        dst: &NodeHandle<N, A>,
        data: A,
    ) -> Option<ArcHandle<N, A>> {
        if !self.check_node_handle(src) || !self.check_node_handle(dst) {
            return None;
        }
        let mut new = Box::new(Arc {
            src: src.ptr,
            dst: dst.ptr,
            data,
            handle: Handle::dummy(),
        });
        let handle = Handle::new(new.as_ref());
        new.handle = sync::Arc::clone(&handle);
        unsafe { &mut *src.ptr.as_ptr() }.out_arcs.push(new);
        unsafe { &mut *dst.ptr.as_ptr() }.in_arcs.push(handle.ptr);
        Some(handle)
    }

    /// Add a new arc from `src` to `dst`, carrying the default data.
    ///
    /// Return a [`Handle`] to the newly created arc.
    ///
    /// See also [`Graph::new_arc_with`].
    pub fn new_arc(
        &mut self,
        src: &NodeHandle<N, A>,
        dst: &NodeHandle<N, A>,
    ) -> Option<ArcHandle<N, A>>
    where
        A: Default,
    {
        self.new_arc_with(src, dst, A::default())
    }

    /// Delete the arc identified by this handle.
    ///
    /// If the handle is not valid anymore (i.e. the arc was already deleted)
    /// then it is simply ignored.
    pub fn delete_arc(&mut self, handle: &ArcHandle<N, A>) {
        let arc = match self.arc_mut(handle) {
            None => return, // this arc no longer exists
            Some(arc) => arc,
        };
        arc.dst_mut().remove_incoming_arc(handle.ptr);
        // the "outgoing version" of the arc must be removed last,
        // because this is what causes the arc to be freed
        arc.src_mut().remove_outgoing_arc(handle.ptr);
        arc.handle.kill();
    }

    /// Delete the node identified by this handle.
    ///
    /// If the handle is not valid anymore (i.e. the node was already deleted)
    /// then it is simply ignored.
    pub fn delete_node(&mut self, handle: &NodeHandle<N, A>) {
        let node = match self.node_mut(handle) {
            None => return, // this node no longer exists
            Some(node) => node,
        };
        let mut in_arcs = vec![];
        std::mem::swap(&mut node.in_arcs, &mut in_arcs);
        for mut aptr in in_arcs {
            unsafe { aptr.as_ref() }.handle.kill();
            let aptr2 = aptr; // copy required by the borrow checker
            let src = unsafe { aptr.as_mut() }.src_mut();
            src.remove_outgoing_arc(aptr2);
        }
        let mut out_arcs = vec![];
        std::mem::swap(&mut node.out_arcs, &mut out_arcs);
        for mut arc in out_arcs {
            arc.handle.kill();
            let ptr = NonNull::from(arc.as_ref());
            let dst = arc.dst_mut();
            dst.remove_incoming_arc(ptr);
        }
        let ptr = node as *const Node<N, A>;
        node.handle.kill();
        match self
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n.as_ref() as *const Node<N, A> == ptr)
        {
            Some((i, _)) => self.nodes.swap_remove(i),
            None => unreachable!("to-be-deleted node could not be found in graph"),
        };
    }

    /// Apply the given [`ChangeList`] to this graph.
    ///
    /// NB: some operations in the [`ChangeList`] may fail,
    /// for example if the handles passed to it are not valid anymore.
    /// Failing operations are simply ignored,
    /// following operations are applied nonetheless.
    pub fn apply(&mut self, changelist: ChangeList<N, A>) {
        changelist.apply_to(self)
    }

    fn check_node_handle(&self, handle: &NodeHandle<N, A>) -> bool {
        handle.alive.load(Ordering::SeqCst)
            && unsafe { handle.ptr.as_ref() }.graph.as_ptr() as *const Self == self
    }

    fn check_arc_handle(&self, handle: &ArcHandle<N, A>) -> bool {
        handle.alive.load(Ordering::SeqCst)
            && unsafe { handle.ptr.as_ref() }.src().graph.as_ptr() as *const Self == self
    }
}

//

impl<N, A> Node<N, A> {
    /// The data associated to this node.
    pub fn data(&self) -> &N {
        &self.data
    }

    /// The data associated to this node.
    pub fn data_mut(&mut self) -> &mut N {
        &mut self.data
    }

    /// The number of outgoing arcs of this node.
    pub fn out_degree(&self) -> usize {
        self.out_arcs.len()
    }

    /// Iterate over the outgoing arcs of this node.
    pub fn out_arcs(&self) -> impl Iterator<Item = &Arc<N, A>> {
        self.out_arcs.iter().map(Box::as_ref)
    }

    /// Iterate over the outgoind arcs (as mutable references) of this node.
    pub fn out_arcs_mut(&mut self) -> impl Iterator<Item = &mut Arc<N, A>> {
        self.out_arcs.iter_mut().map(Box::as_mut)
    }

    /// The number of incoming arcs of this node.
    pub fn in_degree(&self) -> usize {
        self.in_arcs.len()
    }

    /// Iterate over the incoming arcs of this node.
    pub fn in_arcs(&self) -> impl Iterator<Item = &Arc<N, A>> {
        self.in_arcs.iter().map(|a| unsafe { a.as_ref() })
    }

    /// Iterate over the incoming arcs (as mutable references) of this node.
    pub fn in_arcs_mut(&mut self) -> impl Iterator<Item = &mut Arc<N, A>> {
        self.in_arcs.iter_mut().map(|a| unsafe { a.as_mut() })
    }

    /// The graph containing this node.
    pub fn graph(&self) -> &Graph<N, A> {
        unsafe { self.graph.as_ref() }
    }

    /// A [`Handle`] to this node.
    pub fn handle(&self) -> NodeHandle<N, A> {
        self.handle.clone()
    }

    /// Utility method for implementing delete methods in [`Graph`].
    /// Remove an arc from its destination node only.
    /// See also [`Node::remove_outgoing_arc`].
    ///
    /// # Pre-condition
    /// Must only be called with an arc that is actually present in self.in_arcs
    fn remove_incoming_arc(&mut self, arc: NonNull<Arc<N, A>>) {
        match self
            .in_arcs
            .iter()
            .enumerate()
            .find(|(_, a)| unsafe { a.as_ref() } as *const Arc<N, A> == arc.as_ptr())
        {
            Some((i, _)) => self.in_arcs.swap_remove(i),
            None => unreachable!("to-be-deleted arc could not be found in destination node"),
        };
    }

    /// Utility method for implementing delete methods in [`Graph`].
    /// Remove an arc from its source node only.
    /// See also [`Node::remove_incoming_arc`].
    ///
    /// # Pre-condition
    /// Must only be called with an arc that is actually present in self.out_arcs
    ///
    /// # Post-condition
    /// The arc will be freed, so it must not be used anymore
    fn remove_outgoing_arc(&mut self, arc: NonNull<Arc<N, A>>) {
        match self
            .out_arcs
            .iter()
            .enumerate()
            .find(|(_, a)| a.as_ref() as *const Arc<N, A> == arc.as_ptr())
        {
            Some((i, _)) => self.out_arcs.swap_remove(i),
            None => unreachable!("to-be-deleted arc could not be found in source node"),
        };
    }
}

impl<N, A> Hash for Node<N, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle().hash(state)
    }
}

impl<N, A> PartialEq for Node<N, A> {
    fn eq(&self, other: &Node<N, A>) -> bool {
        self.handle() == other.handle()
    }
}

impl<N, A> Eq for Node<N, A> {}

//

impl<N, A> Arc<N, A> {
    /// The data associated to this arc.
    pub fn data(&self) -> &A {
        &self.data
    }

    /// The data associated to this arc.
    pub fn data_mut(&mut self) -> &mut A {
        &mut self.data
    }

    /// The source node of this arc.
    pub fn src(&self) -> &Node<N, A> {
        unsafe { self.src.as_ref() }
    }

    /// The source node of this arc.
    pub fn src_mut(&mut self) -> &mut Node<N, A> {
        unsafe { self.src.as_mut() }
    }

    /// The destination node of this arc.
    pub fn dst(&self) -> &Node<N, A> {
        unsafe { self.dst.as_ref() }
    }

    /// The destination node of this arc.
    pub fn dst_mut(&mut self) -> &mut Node<N, A> {
        unsafe { self.dst.as_mut() }
    }

    /// A [`Handle`] to this arc.
    pub fn handle(&self) -> ArcHandle<N, A> {
        self.handle.clone()
    }
}

impl<N, A> Hash for Arc<N, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle().hash(state)
    }
}

impl<N, A> PartialEq for Arc<N, A> {
    fn eq(&self, other: &Arc<N, A>) -> bool {
        self.handle() == other.handle()
    }
}

impl<N, A> Eq for Arc<N, A> {}

//

impl<T> Handle<T> {
    fn new(owner: &T) -> sync::Arc<Handle<T>> {
        sync::Arc::new(Handle {
            ptr: NonNull::from(owner),
            alive: AtomicBool::new(true),
        })
    }
    fn dummy() -> sync::Arc<Handle<T>> {
        sync::Arc::new(Handle {
            ptr: NonNull::dangling(),
            alive: AtomicBool::new(false),
        })
    }
    fn kill(&self) {
        self.alive.store(false, Ordering::SeqCst);
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("ptr", &self.ptr)
            .field("alive", &self.alive)
            .finish()
    }
}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
        self.alive.load(Ordering::SeqCst).hash(state);
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Handle<T>) -> bool {
        self.ptr == other.ptr
            && self.alive.load(Ordering::SeqCst) == other.alive.load(Ordering::SeqCst)
    }
}

impl<T> Eq for Handle<T> {}

impl<'a, T> PartialEq<&'a T> for Handle<T> {
    fn eq(&self, other: &&'a T) -> bool {
        self.ptr.as_ptr() as *const T == *other && self.alive.load(Ordering::SeqCst)
    }
}

//

#[cfg(test)]
mod tests {
    use super::*;

    type TestGraph = Graph<usize, usize>;

    #[test]
    fn empty() {
        let g = TestGraph::new();
        assert!(g.is_empty());
    }

    #[test]
    fn diamond() {
        let mut g = TestGraph::new();
        let n0 = g.new_node();
        let n1 = g.new_node();
        let n2 = g.new_node();
        let n3 = g.new_node();
        let a01 = g.new_arc(&n0, &n1).unwrap();
        let a02 = g.new_arc(&n0, &n2).unwrap();
        let a13 = g.new_arc(&n1, &n3).unwrap();
        let a23 = g.new_arc(&n2, &n3).unwrap();
        assert_eq!(g.len(), 4);
        assert!(!g.is_empty());
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 2);
        assert_eq!(g.node(&n1).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n2).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n2).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n3).unwrap().in_degree(), 2);
        assert_eq!(g.node(&n3).unwrap().out_degree(), 0);
        assert_eq!(g.arc(&a01).unwrap().src().handle(), n0);
        assert_eq!(g.arc(&a01).unwrap().dst().handle(), n1);
        assert_eq!(g.arc(&a02).unwrap().src().handle(), n0);
        assert_eq!(g.arc(&a02).unwrap().dst().handle(), n2);
        assert_eq!(g.arc(&a13).unwrap().src().handle(), n1);
        assert_eq!(g.arc(&a13).unwrap().dst().handle(), n3);
        assert_eq!(g.arc(&a23).unwrap().src().handle(), n2);
        assert_eq!(g.arc(&a23).unwrap().dst().handle(), n3);
    }

    #[test]
    fn delete_node() {
        let mut g = TestGraph::new();
        let n0 = g.new_node();
        let n1 = g.new_node();
        let n2 = g.new_node();
        let n3 = g.new_node();
        let a01 = g.new_arc(&n0, &n1).unwrap();
        let a02 = g.new_arc(&n0, &n2).unwrap();
        let a13 = g.new_arc(&n1, &n3).unwrap();
        let a23 = g.new_arc(&n2, &n3).unwrap();

        g.delete_node(&n2);

        assert!(!g.is_empty());
        assert_eq!(g.len(), 3);
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n2), None);
        assert_eq!(g.node(&n3).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n3).unwrap().out_degree(), 0);
        assert_eq!(g.arc(&a01).unwrap().src().handle(), n0);
        assert_eq!(g.arc(&a01).unwrap().dst().handle(), n1);
        assert_eq!(g.arc(&a02), None);
        assert_eq!(g.arc(&a13).unwrap().src().handle(), n1);
        assert_eq!(g.arc(&a13).unwrap().dst().handle(), n3);
        assert_eq!(g.arc(&a23), None);

        g.delete_node(&n1);

        assert!(!g.is_empty());
        assert_eq!(g.len(), 2);
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n1), None);
        assert_eq!(g.node(&n2), None);
        assert_eq!(g.node(&n3).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n3).unwrap().out_degree(), 0);
        assert_eq!(g.arc(&a01), None);
        assert_eq!(g.arc(&a02), None);
        assert_eq!(g.arc(&a13), None);
        assert_eq!(g.arc(&a23), None);

        g.delete_node(&n3);

        assert!(!g.is_empty());
        assert_eq!(g.len(), 1);
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n1), None);
        assert_eq!(g.node(&n2), None);
        assert_eq!(g.node(&n3), None);
        assert_eq!(g.arc(&a01), None);
        assert_eq!(g.arc(&a02), None);
        assert_eq!(g.arc(&a13), None);
        assert_eq!(g.arc(&a23), None);

        g.delete_node(&n0);

        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(g.node(&n0), None);
        assert_eq!(g.node(&n1), None);
        assert_eq!(g.node(&n2), None);
        assert_eq!(g.node(&n3), None);
        assert_eq!(g.arc(&a01), None);
        assert_eq!(g.arc(&a02), None);
        assert_eq!(g.arc(&a13), None);
        assert_eq!(g.arc(&a23), None);

        // nodes can be deleted twice
        g.delete_node(&n0);
        g.delete_node(&n1);
        g.delete_node(&n2);
        g.delete_node(&n3);
    }

    #[test]
    fn delete_arc() {
        let mut g = TestGraph::new();
        let n0 = g.new_node();
        let n1 = g.new_node();
        let n2 = g.new_node();
        let n3 = g.new_node();
        let a01 = g.new_arc(&n0, &n1).unwrap();
        let a02 = g.new_arc(&n0, &n2).unwrap();
        let a13 = g.new_arc(&n1, &n3).unwrap();
        let a23 = g.new_arc(&n2, &n3).unwrap();

        g.delete_arc(&a13);

        assert!(!g.is_empty());
        assert_eq!(g.len(), 4);
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 2);
        assert_eq!(g.node(&n1).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n2).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n2).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n3).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n3).unwrap().out_degree(), 0);
        assert_eq!(g.arc(&a01).unwrap().src().handle(), n0);
        assert_eq!(g.arc(&a01).unwrap().dst().handle(), n1);
        assert_eq!(g.arc(&a02).unwrap().src().handle(), n0);
        assert_eq!(g.arc(&a02).unwrap().dst().handle(), n2);
        assert_eq!(g.arc(&a13), None);
        assert_eq!(g.arc(&a23).unwrap().src().handle(), n2);
        assert_eq!(g.arc(&a23).unwrap().dst().handle(), n3);

        g.delete_arc(&a02);

        assert!(!g.is_empty());
        assert_eq!(g.len(), 4);
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n2).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n2).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n3).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n3).unwrap().out_degree(), 0);
        assert_eq!(g.arc(&a01).unwrap().src().handle(), n0);
        assert_eq!(g.arc(&a01).unwrap().dst().handle(), n1);
        assert_eq!(g.arc(&a02), None);
        assert_eq!(g.arc(&a13), None);
        assert_eq!(g.arc(&a23).unwrap().src().handle(), n2);
        assert_eq!(g.arc(&a23).unwrap().dst().handle(), n3);

        g.delete_arc(&a23);

        assert!(!g.is_empty());
        assert_eq!(g.len(), 4);
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().in_degree(), 1);
        assert_eq!(g.node(&n1).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n2).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n2).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n3).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n3).unwrap().out_degree(), 0);
        assert_eq!(g.arc(&a01).unwrap().src().handle(), n0);
        assert_eq!(g.arc(&a01).unwrap().dst().handle(), n1);
        assert_eq!(g.arc(&a02), None);
        assert_eq!(g.arc(&a13), None);
        assert_eq!(g.arc(&a23), None);

        g.delete_arc(&a01);

        assert!(!g.is_empty());
        assert_eq!(g.len(), 4);
        assert_eq!(g.node(&n0).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n0).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n1).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n1).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n2).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n2).unwrap().out_degree(), 0);
        assert_eq!(g.node(&n3).unwrap().in_degree(), 0);
        assert_eq!(g.node(&n3).unwrap().out_degree(), 0);
        assert_eq!(g.arc(&a01), None);
        assert_eq!(g.arc(&a02), None);
        assert_eq!(g.arc(&a13), None);
        assert_eq!(g.arc(&a23), None);

        // arcs can be deleted twice
        g.delete_arc(&a01);
        g.delete_arc(&a02);
        g.delete_arc(&a13);
        g.delete_arc(&a23);
    }

    #[test]
    fn foreign_handle() {
        let mut g1 = TestGraph::new();
        let n1 = g1.new_node();
        let a11 = g1.new_arc(&n1, &n1).unwrap();
        let mut g2 = TestGraph::new();
        let n2 = g2.new_node();
        let a22 = g2.new_arc(&n2, &n2).unwrap();

        assert!(g1.node(&n2).is_none());
        assert!(g1.arc(&a22).is_none());
        assert!(g2.node(&n1).is_none());
        assert!(g2.arc(&a11).is_none());
        assert!(g1.new_arc(&n1, &n2).is_none());
        assert!(g1.new_arc(&n2, &n1).is_none());
        assert!(g1.new_arc(&n2, &n2).is_none());
        assert!(g2.new_arc(&n1, &n2).is_none());
        assert!(g2.new_arc(&n2, &n1).is_none());
        assert!(g2.new_arc(&n1, &n1).is_none());
    }

    #[test]
    fn expired_handle() {
        let mut g = TestGraph::new();
        let n1 = g.new_node();
        let n2 = g.new_node();
        let _a12 = g.new_arc(&n1, &n2).unwrap();

        g.delete_node(&n1);
        g.delete_node(&n2);
        assert!(g.new_arc(&n2, &n1).is_none());
        assert!(g.new_arc(&n1, &n2).is_none());
    }

    #[test]
    fn data_mutating_visitor() {
        let mut g = TestGraph::new();
        let n: Vec<_> = (0..7).map(|_| g.new_node()).collect();
        assert_eq!(g.len(), 7);
        g.new_arc(&n[0], &n[1]);
        g.new_arc(&n[0], &n[2]);
        g.new_arc(&n[1], &n[3]);
        g.new_arc(&n[2], &n[3]);
        g.new_arc(&n[3], &n[4]);
        g.new_arc(&n[3], &n[5]);
        g.new_arc(&n[4], &n[6]);
        g.new_arc(&n[5], &n[6]);

        fn inc_node(n: &mut Node<usize, usize>) {
            *n.data_mut() += 1;
            for arc in n.out_arcs_mut() {
                inc_node(arc.dst_mut());
            }
        }
        inc_node(g.node_mut(&n[2]).unwrap());

        for nt in &n[..2] {
            assert_eq!(g.node(nt).unwrap().data(), &0);
        }
        for nt in &n[2..6] {
            assert_eq!(g.node(nt).unwrap().data(), &1);
        }
        for nt in &n[6..] {
            assert_eq!(g.node(nt).unwrap().data(), &2);
        }
    }

    #[test]
    fn changelist_add_and_mutate() {
        let mut g = TestGraph::new();
        let n0 = g.new_node();
        let n1 = g.new_node();
        let a01 = g.new_arc(&n0, &n1).unwrap();

        let mut cl = ChangeList::new();
        let n2 = cl.new_node();
        let n3 = cl.new_node();
        // arc between old nodes
        let _10 = cl.new_arc(&n1, &n0);
        // arc between old and new node
        let a12 = cl.new_arc(&n1, &n2);
        let _31 = cl.new_arc(&n3, &n1);
        // arc between new nodes
        let a23 = cl.new_arc(&n2, &n3);
        // mutate old node
        cl.mutate_node(&n0, |i| *i += 1);
        // mutate new node
        cl.mutate_node(&n2, |i| *i += 1);
        // mutate old arc
        cl.mutate_arc(&a01, |i| *i += 1);
        // mutate new arcs
        cl.mutate_arc(&a12, |i| *i += 1);
        cl.mutate_arc(&a23, |i| *i += 1);
        g.apply(cl);

        let n: Vec<_> = g.nodes().collect();
        let a: Vec<_> = n.iter().flat_map(|n| n.out_arcs()).collect();

        assert_eq!(n[0].in_degree(), 1);
        assert_eq!(n[0].out_degree(), 1);
        assert_eq!(n[0].data(), &1);
        assert_eq!(n[1].in_degree(), 2);
        assert_eq!(n[1].out_degree(), 2);
        assert_eq!(n[1].data(), &0);
        assert_eq!(n[2].in_degree(), 1);
        assert_eq!(n[2].out_degree(), 1);
        assert_eq!(n[2].data(), &1);
        assert_eq!(n[3].in_degree(), 1);
        assert_eq!(n[3].out_degree(), 1);
        assert_eq!(n[3].data(), &0);
        assert_eq!(n.len(), 4);

        assert_eq!(a[0].src(), n[0]);
        assert_eq!(a[0].dst(), n[1]);
        assert_eq!(a[0].data(), &1);
        assert_eq!(a[1].src(), n[1]);
        assert_eq!(a[1].dst(), n[0]);
        assert_eq!(a[1].data(), &0);
        assert_eq!(a[2].src(), n[1]);
        assert_eq!(a[2].dst(), n[2]);
        assert_eq!(a[2].data(), &1);
        assert_eq!(a[3].src(), n[2]);
        assert_eq!(a[3].dst(), n[3]);
        assert_eq!(a[3].data(), &1);
        assert_eq!(a[4].src(), n[3]);
        assert_eq!(a[4].dst(), n[1]);
        assert_eq!(a[4].data(), &0);
        assert_eq!(a.len(), 5);
    }

    #[test]
    fn changelist_delete() {
        let mut g = TestGraph::new();
        let n0 = g.new_node_with(0);
        let n1 = g.new_node_with(1);
        let n2 = g.new_node_with(2);
        let _01 = g.new_arc_with(&n0, &n1, 01).unwrap();
        let a12 = g.new_arc_with(&n1, &n2, 12).unwrap();

        let mut cl = ChangeList::new();
        let n3 = cl.new_node_with(3);
        let n4 = cl.new_node_with(4);
        let a23 = cl.new_arc_with(&n2, &n3, 23);
        let _34 = cl.new_arc_with(&n3, &n4, 34);
        // deleting old node
        cl.delete_node(&n0);
        // deleting new node
        cl.delete_node(&n4);
        // deleting old arc
        cl.delete_arc(&a12);
        // deleting old arc
        cl.delete_arc(&a23);
        g.apply(cl);

        let mut n: Vec<_> = g.nodes().collect();
        n.sort_by_key(|n| n.data());
        let a: Vec<_> = n
            .iter()
            .flat_map(|n| n.out_arcs().map(Arc::handle))
            .collect();

        assert_eq!(n[0].data(), &1);
        assert_eq!(n[0].in_degree(), 0);
        assert_eq!(n[0].out_degree(), 0);
        assert_eq!(n[1].data(), &2);
        assert_eq!(n[1].in_degree(), 0);
        assert_eq!(n[1].out_degree(), 0);
        assert_eq!(n[2].data(), &3);
        assert_eq!(n[2].in_degree(), 0);
        assert_eq!(n[2].out_degree(), 0);
        assert_eq!(n.len(), 3);

        assert_eq!(a.len(), 0);
    }

    #[test]
    fn changelist_invalid_handles() {
        let mut g = TestGraph::new();
        let n0 = g.new_node_with(0);
        let n1 = g.new_node_with(1);
        let n2 = g.new_node_with(2);
        let _01 = g.new_arc_with(&n0, &n1, 01).unwrap();
        let a12 = g.new_arc_with(&n1, &n2, 12).unwrap();
        g.delete_node(&n0);
        g.delete_arc(&a12);

        let mut cl = ChangeList::new();
        let n3 = cl.new_node_with(3);
        let n4 = cl.new_node_with(4);
        let a23 = cl.new_arc_with(&n2, &n3, 23);
        let _34 = cl.new_arc_with(&n3, &n4, 34);
        cl.delete_node(&n4);
        cl.delete_arc(&a23);
        // adding arcs between missing nodes
        let a03 = cl.new_arc(&n0, &n3);
        let a14 = cl.new_arc(&n1, &n4);
        let a04 = cl.new_arc(&n0, &n4);
        // removing missing arcs (deleted before)
        cl.delete_arc(&a12);
        cl.delete_arc(&a23);
        // removing missing arcs (could not be created)
        cl.delete_arc(&a03);
        cl.delete_arc(&a14);
        cl.delete_arc(&a04);
        // removing missing nodes
        cl.delete_node(&n0);
        cl.delete_node(&n4);
        g.apply(cl);

        let mut n: Vec<_> = g.nodes().collect();
        n.sort_by_key(|n| n.data());
        let a: Vec<_> = n.iter().flat_map(|n| n.out_arcs()).collect();

        assert_eq!(n[0].data(), &1);
        assert_eq!(n[0].in_degree(), 0);
        assert_eq!(n[0].out_degree(), 0);
        assert_eq!(n[1].data(), &2);
        assert_eq!(n[1].in_degree(), 0);
        assert_eq!(n[1].out_degree(), 0);
        assert_eq!(n[2].data(), &3);
        assert_eq!(n[2].in_degree(), 0);
        assert_eq!(n[2].out_degree(), 0);
        assert_eq!(n.len(), 3);

        assert_eq!(a.len(), 0);
    }

    #[test]
    fn changelist_during_iteration() {
        let mut g = TestGraph::new();
        let n: Vec<_> = (0..8).map(|i| g.new_node_with(i)).collect();
        assert_eq!(g.len(), 8);
        for i in 0..8 {
            g.new_arc(&n[i], &n[i % 8]);
        }

        let mut cl = ChangeList::new();
        for n in g.nodes().filter(|n| n.data() % 2 == 0) {
            let nn = cl.new_node();
            cl.new_arc(&n.handle(), &nn);
        }
        g.apply(cl);
        assert_eq!(g.len(), 12);
    }
}

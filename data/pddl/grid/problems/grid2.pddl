;; 90 states
;; 3x3 grid with 1 key
(define (problem medium)
  (:domain grid)

  (:objects node0-0 node0-1 node0-2
  		node1-0 node1-1 node1-2
		node2-0 node2-1 node2-2
	    triangle diamond square circle
	    key0)

  (:init (place node0-0) (place node0-1) (place node0-2)
	 (place node1-0) (place node1-1) (place node1-2)
	 (place node2-0) (place node2-1) (place node2-2)
	 (shape triangle) (shape diamond) (shape square) (shape circle)
	 (key key0)
     (holding key0)
	 (conn node0-0 node0-1) (conn node0-1 node0-0)
	 (conn node1-0 node1-1) (conn node1-1 node1-0)
	 (conn node0-0 node1-0) (conn node1-0 node0-0)
	 (conn node0-1 node1-1) (conn node1-1 node0-1)
     (conn node0-1 node0-2) (conn node0-2 node0-1)
     (conn node1-1 node1-2) (conn node1-2 node1-1)
     (conn node0-2 node1-2) (conn node1-2 node0-2)
	 (conn node1-0 node2-0) (conn node2-0 node1-0)
	 (conn node1-1 node2-1) (conn node2-1 node1-1)
	 (conn node1-2 node2-2) (conn node2-2 node1-2)
	 (conn node2-0 node2-1) (conn node2-1 node2-0)
	 (conn node2-1 node2-2) (conn node2-2 node2-1)
	 (open node0-0) (open node0-1) (open node0-2)
	 (open node1-0) (open node1-1) (open node1-2)
	 (open node2-0) (open node2-1) (open node2-2)
	 (key-shape key0 triangle)
	 (at-robot node0-0))
  (:goal (at key0 node2-2))
  )

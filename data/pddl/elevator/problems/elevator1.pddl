;; 36 states
;; 2 passengers and 4 floors

(define (problem mixed-f4-p2-u0-v0-g0-a0-n0-A0-B0-N0-F0-r4)
   (:domain miconic)
   (:objects p0 p1 - passenger
             f0 f1 f2 f3 - floor)


(:init
(above f0 f1)
(above f1 f2)
(above f2 f3)

(origin p0 f3)
(destin p0 f2)

(origin p1 f2)
(destin p1 f0)

(lift-at f0)
)

(:goal (and 
(served p0)
(served p1)
))
)
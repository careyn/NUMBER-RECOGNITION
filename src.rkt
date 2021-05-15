(require 2htdp/batch-io)
(require 2htdp/image)
(require 2htdp/universe)
(define LOF-EMPTY empty)

; Exercise 1

(define-struct node [l r])
; A LeafyTree is one of:
; - "leaf"
; - (make-node LeafyTree LeafyTree)
; Interpretation: a binary tree that ends in a leaf

(define LEAFYTREE-0 "leaf")
(define LEAFYTREE-1 (make-node LEAFYTREE-0 LEAFYTREE-0))
(define LEAFYTREE-2 (make-node LEAFYTREE-1 LEAFYTREE-1))
(define LEAFYTREE-3 (make-node LEAFYTREE-2 LEAFYTREE-2))
(define LEAFYTREE-4 (make-node LEAFYTREE-3 LEAFYTREE-3))

(define (lf-temp lf)
  (...
   (cond [(string? lf) ...]
         [(node? lf)
          (lf-temp (node-l lf))...
          (lf-temp (node-r lf))...])))

; An LR is one of:
; - "left"
; - "right"
; Interpretation: directions one can take in a LeafyTree

(define L "left")
(define R "right")

(define (lr-temp lr)
  (...
   (cond [(string=? lr L) ...]
         [(string=? lr R) ...])))

; A Path is a [List-of LR]

(define PATH-0 empty)
(define PATH-1 (list L))
(define PATH-2 (list R))
(define PATH-3 (list L R L R))

(check-expect (paths LEAFYTREE-0) (list empty))
(check-expect (paths LEAFYTREE-3)(list
                                  (list L L L)
                                  (list L L R)
                                  (list L R L)
                                  (list L R R)
                                  (list R L L)
                                  (list R L R)
                                  (list R R L)
                                  (list R R R)))
; paths : LeafyTree -> [List-of Paths]
; Finds all the root to leaf paths of a given tree
(define (paths lf)
  (cond [(string? lf) (list empty)]
        [(node? lf) (append (pathmaker "left" (paths (node-l lf)))
                            (pathmaker "right" (paths (node-r lf))))]))
; pathmaker : LeafyTree -> [List-of Paths]
; adds together the path as the above program runs
(check-expect (pathmaker R (list PATH-0 PATH-1 PATH-3)) (list
                                                         (list R)
                                                         (list R L)
                                                         (list R L R L R)))
                                                         
(define (pathmaker direction path)
  (map (λ (x) (cons direction x)) path))

; Exercise 2

; An Operation is one of:
; - "+"
; - "*"
; Interpretation: a commutative operation
(define PLUS "+")
(define TIMES "*")

(define (o-temp o)
  (...
   (cond [(string=? o PLUS)...]
         [(string=? o TIMES)...])))

; An AExp (Arithmetic Expression) is one of:
; - Number
; - (cons Operation [List-of AExp])

(define AEXP-0 0)
(define AEXP-1 (cons PLUS (list 1 2 3 4)))
(define AEXP-2 (cons TIMES (list 1 2 3 4)))
(define AEXP-3 (cons TIMES (list (cons PLUS (list 1 2 3)) 5)))
(cons TIMES (list 1 2 (list PLUS (list 1 2))))
(define (aexp-temp a)
  (...
   (cond [(number? a) ...]
         [(cons? a)
          (...
           (o-temp (first a))...
           ...(rest a)...)]))) 

; eval : AExp -> Number
; consumes an AExp and evaluates it to a single number using the giver operators
(check-expect (eval AEXP-0) 0)
(check-expect (eval AEXP-1) 10)
(check-expect (eval AEXP-2) 24)
(check-expect (eval AEXP-3) 30)
(define (eval a)
  (cond [(number? a) a]
        [(list? (second a)) 
         (check-operation (cons (first a) (map eval (rest a))))]
        [(string? (first a)) (check-operation a)]))  

; check-operation : (cons Operation [List-of AExp]) -> Number
; Determines the operation and maps it to the list
(check-expect (check-operation AEXP-1) 10)
(check-expect (check-operation AEXP-2) 24)
(define (check-operation l)
  (cond [(empty? (first l)) l]
        [(string=? (first l) PLUS) (foldr + 0 (rest l))]
        [(string=? (first l) TIMES) (foldr * 1 (rest l))]))

; Exercise 3

;; training-fnames : Nat -> [List-of String]
;; Produce the names of all training files given the number of examples per digit
(check-expect (training-fnames 0) empty)
(check-expect
 (training-fnames 3)
 (list "train/d_1_0.txt" "train/d_2_0.txt" "train/d_3_0.txt"
       "train/d_1_1.txt" "train/d_2_1.txt" "train/d_3_1.txt"
       "train/d_1_2.txt" "train/d_2_2.txt" "train/d_3_2.txt"
       "train/d_1_3.txt" "train/d_2_3.txt" "train/d_3_3.txt"
       "train/d_1_4.txt" "train/d_2_4.txt" "train/d_3_4.txt"
       "train/d_1_5.txt" "train/d_2_5.txt" "train/d_3_5.txt"
       "train/d_1_6.txt" "train/d_2_6.txt" "train/d_3_6.txt"
       "train/d_1_7.txt" "train/d_2_7.txt" "train/d_3_7.txt"
       "train/d_1_8.txt" "train/d_2_8.txt" "train/d_3_8.txt"
       "train/d_1_9.txt" "train/d_2_9.txt" "train/d_3_9.txt"))
(define (training-fnames examples-per-digit)
  (foldr (λ (digit sofar)
           (append (map (λ (ex) (generate-training-file-name ex digit))
                        (build-list examples-per-digit add1))
                   sofar))
         empty
         (build-list 10 identity)))

;; generate-training-file-name : Nat Nat -> String
;; Generate a training file name for the given digit and example id
(check-expect (generate-training-file-name 1 0) "train/d_1_0.txt")
(check-expect (generate-training-file-name 7 3) "train/d_7_3.txt")
(define (generate-training-file-name exid digit)
  (string-append "train/d_" (number->string exid) "_" (number->string digit) ".txt"))

;; read-lolon : String -> [List-of [List-of Number]]
;; Read the numbers from the given file
(check-expect (read-lolon "numbers.txt")
              (list
               (list 0 0 0 0)
               (list 1 2 3 4)
               (list 2 1 0 -0.5)))

;; A Bitmap is one of:
;; - empty
;; - (cons ListOfFeatures Bitmap)
;; and represents a grid of grayscale pixels

(define BITMAP-EMPTY empty)
(define BITMAP-ZERO
  (list (list 255 255 255)
        (list 255 0 255)
        (list 255 255 255)))

;; bitmap-template : Bitmap -> ???
(define (bitmap-template b)
  (cond
    [(empty? b) ...]
    [(cons? b)
     (...  (lof-template (first b)) ...
           (bitmap-template (rest b)) ...)]))

(define PIXEL-SIZE 10)
(define PIXEL-BLACK (square PIXEL-SIZE "solid" "black"))
(define PIXEL-WHITE (square PIXEL-SIZE "solid" "white"))

;; bitmap->image : Bitmap -> Image
;; Returns the visualization of the given bitmap
;(check-expect (bitmap->image BITMAP-EMPTY) empty-image)
~embed:1~
(define (bitmap->image bm)
  (foldr (λ (row sofar) (above (bitmap-row->image row) sofar))
         empty-image
         bm))

;; bitmap-row->image : [List-of Feature] -> Image
;; Produce an image of each feature in a line
;(check-expect (bitmap-row->image LOF-EMPTY) empty-image)
;(check-expect (bitmap-row->image (list 0 255)) (beside PIXEL-WHITE PIXEL-BLACK))
(define (bitmap-row->image row)
  (foldr (λ (f sofar) (beside (feature->image f) sofar))
         empty-image
         row))

;; feature->image : Feature -> Image
;; Produce an image of a pixel with the given color
;(check-expect (feature->image 0) PIXEL-WHITE)
;(check-expect (feature->image 255) PIXEL-BLACK)
(define (feature->image f)
  (local [(define flipped (- 255 (string->number f)))]
    ;; Okay to use a helper to find the correct color
    (square PIXEL-SIZE "solid" (make-color flipped flipped flipped))))


(define (read-lolon fpath)
  (map (λ (los) (map string->number los)) (read-words/line fpath)))

(define LIST-MISMATCH "Lists must be the same size")
;; map-2list : (X Y Z) [X Y -> Z] [List-of X] [List-of Y] -> [List-of Z]
;; Apply the given function to each pair of elements
(check-expect (map-2list string=? empty empty) empty)
(check-error (map-2list * (list 10) (list 7 2)))
(check-error (map-2list - (list 7 2) (list 10)))
(check-expect (map-2list + (list 1 2) (list 3 4)) (list 4 6))
(define (map-2list transformer l1 l2)
  (cond [(and (empty? l1) (empty? l2)) empty]
        [(and (empty? l1) (cons? l2)) (error LIST-MISMATCH)]
        [(and (cons? l1) (empty? l2)) (error LIST-MISMATCH)]
        [(and (cons? l1) (cons? l2))
         (cons (transformer (first l1) (first l2))
               (map-2list transformer (rest l1) (rest l2)))]))

;; smallest-of-list-by-f : (X) [X -> Number] [NEList-of X] -> X
;; Find the element of the list that minimizes the given function
(check-expect (smallest-of-list-by-f string-length (list "hello")) "hello")
(check-expect
 (smallest-of-list-by-f
  length
  (list (list 1 2 3) (list 100) (list -1000 -99 -1 0) (list 2)))
 (list 100))
(define (smallest-of-list-by-f transformer nelox)
  (foldr (λ (element sofar)
           (if (<= (transformer element) (transformer sofar)) element sofar))
         (first nelox)
         (rest nelox)))

;; next-index : (X) [List-of X] Nat -> Nat
;; Produces the next index in the list
(check-expect (next-index (list 1 2 3) 0) 1)
(check-expect (next-index (list "hello" "world") 1) 0)
(define (next-index lox index)
  ;; Okay to use if/cond instead of modulo
  (modulo (add1 index) (length lox)))

;; flatten : (X) [List-of [List-of X]] -> [List-of X]
;; Flatten the nested list into a single-level list
(check-expect (flatten empty) empty)
(check-expect
 (flatten (list (list 1 2 3) (list "hello" "world")))
 (list 1 2 3 "hello" "world"))
(define (flatten lox)
  (foldr append empty lox))

;; prev-index : (X) [List-of X] Nat -> Nat
;; Produces the previous index in the list
(check-expect (prev-index (list 1 2 3) 0) 2)
(check-expect (prev-index (list "hello" "world") 1) 0)
(define (prev-index lox index)
  ;; Okay to use if/cond instead of modulo
  (modulo (sub1 index) (length lox)))

;; return-former : Any Any -> Any
;; Returns the first argument
(check-expect (return-former 1 "hello") 1)
(check-expect (return-former true false) true)
(define (return-former a b) a)


(define-struct bitstate [l ref])
; bitstate represents the current state of the reference through all the files
; and is a (make-bitstate Bitmap Number)
(define B0 (make-bitstate (list (list "0" "0" "0") (list "0" "0" "0") (list "0" "0" "0")) 0))
(define B1 (make-bitstate (list (list "1" "1" "1") (list "1" "1" "1") (list "1" "1" "1")) 1))

(define (bs-temp bs)
  (...
   ...(bitstate-l bs)...
   ...(bitsate-ref bs)...))

; mnist : Number String -> WorldState
; visualizes the comparisons between test and training and best match
(define (mnist num str)
  (local [(define listo (map read-words/line (training-fnames num)))]
    (big-bang (make-bitstate (first listo) 0)
      [to-draw (λ (c)
                 (beside
                  (above (above (text "TEST IMAGE" 24 "red")
                                (bitmap->image (read-words/line str)))
                         (above
                          (text (string-append
                                 "BEST MATCH: "
                                 (number->string
                                  (foldr min 5956785678567856 (map (λ (x) (euclidian-distance
                                                                           (read-words/line str) x))
                                                                   listo))))
                                24 "blue")
                          (bitmap->image (determine-bitmap
                                          (foldr min 3945723948572938475
                                                 (map (λ (x)
                                                        (euclidian-distance
                                                         (read-words/line str) x)) listo)) listo str)
                                         )))
                  (above
                   (text (string-append "TRAINING: "
                                        (number->string (euclidian-distance
                                                         (read-words/line str) (bitstate-l c))))
                         24 "purple")(bitmap->image (bitstate-l c)))))]
      [on-key (λ (x ke) (cond [(string=? ke "right")
                               (make-bitstate (list-ref listo (next-index listo (bitstate-ref x)))
                                              (add1 (bitstate-ref x)))]
                              [(string=? ke "left")
                               (make-bitstate (list-ref listo (prev-index listo (bitstate-ref x)))
                                              (sub1 (bitstate-ref x)))]
                              [else x]))])))    

; euclidian-distance : [List-of Numbers] [List-of Numbers] -> Number
; Determines the euclidian distance between two bitmaps
(check-expect (euclidian-distance (list (list "0" "0" "0") (list "0" "0" "0") (list "0" "0" "0"))
                                  (list (list "0" "0" "0") (list "0" "0" "0") (list "0" "0" "0"))) 0)
(check-expect (euclidian-distance (list (list "1" "1" "1") (list "1" "1" "1") (list "1" "1" "1"))
                                  (list (list "0" "0" "0") (list "0" "0" "0") (list "0" "0" "0"))) 3)
(define (euclidian-distance test training)
  (sqrt (foldr + 0 (map-2list (λ (x y) (sqr (- x y)))  (map string->number (flatten test))
                              (map string->number (flatten training))))))


; determine-bitmap : Number Bitmap String -> Bitmap
; Determines which of the bitmaps has the same euclidian distance as the min euclidian distance
; check-expects wont work because it references big-bang data and specific files
(define (determine-bitmap num l str)
  (cond [(empty? l) l]
        [(equal? num (euclidian-distance (read-words/line str) (first l))) (first l)]
        [else (determine-bitmap num (rest l) str)]))

; (mnist 30 "test/d_6100_1.txt")

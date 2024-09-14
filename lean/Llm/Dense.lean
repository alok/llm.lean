import LinearAlgebra.Vector
import Llm.Matmul
import Llm.Softmax

/-- This is necessary to avoid lotsa headaches.-/
local instance : Hashable Float where
  hash x :=  x.toUInt64

/-- A dense layer. I corresponds to the rows of a matrix, O to the columns.-/
structure Dense (I O: ℕ) where
  /--weights, default is all zeros -/
  W : Vector O (Vector I Float) := 0
  /--bias, default is all ones-/
  b : Vector O Float := 1
deriving Hashable, Inhabited

instance : Repr (Dense I O) where
  reprPrec self _ := s!"Weight: {self.W}\nBias: {self.b}"

instance : ToString (Dense I O) where
  toString self := s!"{repr self}"


def Dense.forward_unbatched (self : Dense I O) (xs: Vector I Float) : Vector O Float :=
  (self.W * xs) + self.b

#eval {:Dense 1 3}.forward_unbatched !v[3]
-- instance : CoeFun (Dense I O) (fun _ => Vector I Float -> Vector O Float) where
--   coe f := f.forward_unbatched

-- TODO(alok): ask cursor composer to define matrix abbrev and use that so we stop messing up matrix dim order



def Dense.forward (self : Dense I O) (xs: Vector B (Vector I Float)) : Vector B (Vector O Float) :=
  xs.map self.forward_unbatched


def Dense.backward
  (self: Dense I O)
  (inp: Vector B (Vector I Float))
  (dout: Vector B (Vector O Float))
:
  -- gradients for dense layer
  Dense I O ×
  -- dinp
  (Vector B (Vector I Float ) )
:=

  -- B x O x 1
  -- unsqueeze dout at dim 2
  let dout': Vector B (Vector O (Vector 1 Float))  := dout.map (·.map Vector.singleton)

  -- B x 1 x I
  -- unsqueeze inp at dim 1
  let inp_t : Vector B (Vector 1 (Vector I Float)) := inp.map Vector.singleton

  let d_bias : Vector O Float  := dout.sum -- equivalent to dotting with vector of all 1s

  let W_t := self.W.transpose

  let d_W : Vector O (Vector I Float)  := (matmul_batched dout' inp_t).sum

  let d_inp := matmul_batched (Vector.replicate B W_t) dout'
  let d_inp' := d_inp.map (·.transpose[0])

  let dense_gradients := Dense.mk d_W d_bias

  (dense_gradients, d_inp')

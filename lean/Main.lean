import LinearAlgebra.Vector
import Llm.Dense

-- Define constants and helper functions first
def KARPATHY_CONSTANT : Float := 3e-4

structure Config where
  numEpochs: ℕ := 1500
  lr: Float := KARPATHY_CONSTANT
  batchSize: ℕ := 4
  inputSize: ℕ := 2
deriving Repr,Inhabited

def CONFIG : Config := {}
def B := CONFIG.batchSize
def I := CONFIG.inputSize

-- TODO this is to get n * a to work for `a: Float`
local instance [Add α] [Zero α] : HMul Nat α α where
  hMul n a := Id.run do
    let mut result := 0
    for i in [0:n/2] do
      result := result + a + a
    if n % 2 == 1 then
      result := result + a
    return result

#eval 2 * 3.3
/-- naive exponentiation by squaring-/
local instance [Mul α] [One α] : HPow α Nat α where
  hPow a n := Id.run do
    let mut (base, exp, result) := (a, n, 1)
    while exp > 0 do
      if exp % 2 == 1 then
        result := result * base
      base := base * base
      exp := exp / 2
    return result

-- Define data and model
def DATA: Vector 4 (Vector 2 Float) :=
  !v[
    !v[1, 0],
    !v[2, 0],
    !v[3, 1],
    !v[4, 1]
  ]

def TARGETS: Vector B Float := !v[0,1,2,3]

def model : Dense I 1 := default

-- Define loss functions
def mse (self other : Vector n Float) :=
  (self.zip other |>.map (fun (s, o) => (s - o) ^ 2)).sum / n.toFloat



def mse.backward (self other : Vector n Float) :=
  (self.zip other |>.map (fun (s, o) => 2 * (s - o)))

-- Define training loop
def trainLoop (config: Config) (network: Dense I 1) (data: Vector B (Vector I Float)) (targets: Vector B Float) := Id.run do
  dbg_trace s!"targets: {targets}"
  let mut network := network
  let mut loss := Float.inf
  for epoch in [0:config.numEpochs] do
    let predictions := network.forward data |>.squeeze

    loss := mse predictions targets

    let dloss := mse.backward predictions targets
    if epoch % 500 == 0 then
      dbg_trace s!"predictions: {predictions}"
      dbg_trace s!"loss: {loss}"
      dbg_trace s!"dloss: {dloss}"
    let (denseGrad, _) := network.backward data dloss.unsqueeze

    -- minus because we want to minimize
    let new_model := Dense.mk (network.W - config.lr * denseGrad.W) (network.b - config.lr * denseGrad.b)
    network := new_model
  return (network, loss)

-- Main function
def main : IO Unit := do
  let config : Config := {}  -- Use default values
  let (new_model, loss) := trainLoop {config with lr:= 1e-3} model DATA TARGETS
  IO.println <| s!"Final loss: {loss}"
  IO.println <| s!"Final model: {new_model}"

#eval main

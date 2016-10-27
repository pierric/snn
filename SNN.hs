{-# LANGUAGE ScopedTypeVariables #-}
module SNN where
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import System.Random.MWC
import System.Random.MWC.Distributions
import Control.Monad.ST
import Data.List

import Debug.Trace

data NN = NN {
    network   :: [LL],
    activate  :: Double -> Double,  -- activate function
    activate' :: Double -> Double   -- derivation of activate function
}

data LL = LL {
    -- column i is the weights for neuron i of the layer
    weights :: Matrix Double,
    -- bias of each neuron+
    biases  :: Vector Double
}

newNN :: [Int] -> NN
newNN desc = NN net relu relu'
  where
    ws = runST $ do
        rseed <- create
        let buildMatrix (nr, nc) = do
                vals <- sequence (replicate (nr*nc) (normal 0 0.01 rseed))
                return $ (nr >< nc) vals
        mapM buildMatrix $ zip desc (tail desc)
    bs = map (konst 1) (tail desc)
    net = zipWith LL ws bs

-- forward propagation and trace the (before-activation of each neuron, after-
-- activation of each neuron) of each layer (in reverse order)
forwardWithTrace :: Vector Double -> NN -> [(Vector Double, Vector Double)]
forwardWithTrace input (NN{network=net,activate=af}) = walk input net []
  where
    walk _ [] tr = tr
    walk i (LL{weights=w,biases=b}:nn) tr =
      let sv = (i <# w) `add` b
          av = cmap af sv
      in walk av nn ((sv,av):tr)

forward i n = snd $ head $ forwardWithTrace i n

relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y

rate = 0.008

-- from input data and expected output, the NN evolves.
learn :: (Vector Double, Vector Double) -> NN -> NN
learn (inp, out) nn@(NN{network=net,activate'=af'}) =
  let (bn, an):ls = forwardWithTrace inp nn
      -- calculate the delta_n for the last layer
      -- w.r.t. the expected output
      dn = (zipVectorWith cost' an out) `hadamad` (cmap af' bn)
      ds :: [Vector Double]
      ds = dn : zipWith3 backPropagation (reverse $ map weights net) ds (map fst ls)
      backPropagation wia1 dia1 bi =
          (wia1 #> dia1) `hadamad` (cmap af' bi)
      -- apply the learning rate
      ds' = map (scale (negate rate)) ds
      -- matrix update for each layer (reversed order)
      md :: [Matrix Double]
      md = zipWith outer (map snd ls ++ [inp]) ds'
      -- bias update for each layer (reversed order)
      bd :: [Vector Double]
      bd = ds'
      -- update function for each layer
      upd dw db (LL{weights=w, biases=b}) = LL{weights=w `add` dw, biases=b `add` db}
  in --traceShow (head md, head bd) $
     nn{network=zipWith3 upd (reverse md) (reverse bd) net}
  where
    hadamad = zipVectorWith (*)

module SNN where
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import System.Random.MWC
import Control.Monad.ST
import Data.List

data NN = NN {
    network   :: [LL],
    activate  :: Double -> Double,  -- activate function
    activate' :: Double -> Double   -- derivation of activate function
}
-- column i is the weights for neuron i of the layer
type LL = Matrix Double

newNN :: [Int] -> NN
newNN desc = NN net relu relu'
  where
    net = runST $ do
        rseed <- create
        let norm stdev = do
                x1 <- uniform rseed
                x2 <- uniform rseed
                return $ stdev * sqrt (-2 * log x1) * cos (2 * pi * x2)
            buildMatrix (nr, nc) = do
                vals <- sequence (replicate (nr*nc) (norm 0.01))
                return $ (nr >< nc) vals
        mapM buildMatrix $ zip desc (tail desc)

-- forward propagation and trace the (before-activation of each neuron, after-
-- activation of each neuron) of each layer (in reverse order)
forwardWithTrace :: Vector Double -> NN -> [(Vector Double, Vector Double)]
forwardWithTrace input (NN{network=net,activate=af}) = walk input net []
  where
    walk _ [] tr = tr
    walk i (w:nn) tr =
      let sv = i <# w
          av = cmap af sv
      in walk av nn ((sv,av):tr)

forward i n = snd $ head $ forwardWithTrace i n

relu = max 0
relu' x | x < 0     = 0
        | otherwise = 1

cost' a y | y == 1 && a >= y = 0
          | otherwise        = a - y

rate = 0.002

-- from input data and expected output, the NN evolves.
learn :: (Vector Double, Vector Double) -> NN -> NN
learn (inp, out) nn@(NN{network=net,activate'=af'}) =
  let (sn, an):tr = forwardWithTrace inp nn
      -- calculate the delta_n for the last layer
      -- w.r.t. the expected output
      dn = (zipVectorWith cost' an out) `hadamad` (cmap af' sn)
      ds :: [Vector Double]
      ds = dn : zipWith3 backPropagation net ds tr
      backPropagation wia1 dia1 (si,_) = (wia1 #> dia1) `hadamad` (cmap af' si)
      -- update for each layer (reversed order)
      md :: [Matrix Double]
      md = zipWith outer (map snd tr ++ [inp]) ds
  in nn{network=zipWith add (reverse md) net}
  where
    hadamad = zipVectorWith (*)

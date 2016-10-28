module Main where

import SNN
import Parser
import Numeric.LinearAlgebra (Vector, vector)

import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)
import Text.PrettyPrint.Boxes
import Numeric.LinearAlgebra.Data

main = mnistMain

mnistMain = do
    let nn = newNN [784, 30, 10]
    putStrLn "Load training data."
    dataset <- map preprocess . uncurry zip <$> trainingData
    putStrLn "Load test data."
    testset <- uncurry zip <$> testData
    putStrLn "Test"
    let nns = iterate (online dataset) nn
        nn' = nns!!100
    nn' `seq` dotest nn' testset
    return ()

online dataset nn = foldl' (flip learn) nn dataset

preprocess :: (Image, Label) -> (Vector Double, Vector Double)
preprocess (i, l) =
    let iv = vector $ map ((/256) . fromIntegral) i
        ll = fromIntegral l
        ov = vector (replicate ll 0 ++ [1] ++ replicate (9-ll) 0)
    in (iv, ov)

postprocess :: Vector Double -> Label
postprocess = fst . maximumBy cmp . zip [0..] . toList
  where cmp a b = compare (snd a) (snd b)

dotest :: NN -> [(Image, Label)] -> IO ()
dotest nn t = do
    let result = map (postprocess . flip forward nn . fst . preprocess) t
        expect = map snd t
        (co,wr)= partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)

-- printO = mapM_ (\(v, a) -> putStrLn $ show v ++ " : " ++ show a) . zip [0..] . toList

dumpMain = do
    let nn = newNN [2,2,2]
        ds = [([0,0],[1,0]), ([0,1],[0,1]), ([1,0],[0,1]), ([1,1],[1,0])]
        inp = map (\(l1, l2) -> (vector l1, vector l2)) ds
        pass n0 = foldl' (flip learn) n0 inp
        ns = iterate pass nn
    dumpNN nn
    dumpNN (ns !! 1000)
    dumpNN (ns !! 2000)
    dumpNN (ns !! 3000)
    dumpNN (ns !! 4000)

stepMain = do
  let l1 = (2><2) [-0.0213,  0.0144, 0.0014,  -0.0001]
      b1 = vector [0.0722, 0.0632]
      l2 = (2><2) [ -0.2354,  -0.2459, -0.2841,  -0.2772]
      b2 = vector [0.5349, 0.5349]
      nt = [LL l1 b1, LL l2 b2]
      nn = NN { network = nt, activate = relu, activate' = relu' }
      step1 = learn (vector [0,0], vector [1,0])
      step2 = learn (vector [0,1], vector [0,1])
      step3 = learn (vector [1,0], vector [0,1])
      step4 = learn (vector [1,1], vector [0,0])
      nns = iterate step2 nn
  dumpNN nn
  dumpNN $ step4 $ step3 $ step2 $ step1 nn

dumpNN nn@(NN{network=net}) = do
    putStrLn $ render $ hsep 6 top [dump, apply]
  where
    dump  = vcat left [text "=========== DUMP NN ===========", hsep 5 center1 (map prL net)]
    apply= vcat left [text "===========  APPLY  ===========", body]
    body = text $ show $ forward (vector [0,1]) nn

prL l = hsep 1 center1 [prM (weights l), text "+", prB (biases l)]
prM m = vcat right (map prV (toRows m))
prV v = hsep 2 center1 (map prE (toList v))
prB b = vcat right (map prE (toList b))
prE f = text $ printf "%.4f" f

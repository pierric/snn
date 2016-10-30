module Main where

import SNN
import Parser

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Data.List(foldl',partition,maximumBy)
import Text.Printf (printf)
--import Text.PrettyPrint.Boxes
--import System.Random.MWC
--import Data.Array.IO
--import Control.Monad

main = mnistMain

mnistMain = makeMNIST >>= dotest

makeMNIST = do
    let nn = newNN [784, 30, 10]
    putStrLn "Load training data."
    dataset <- uncurry zip <$> trainingData
    putStrLn "Load test data."
    let nns = iterate (online dataset) nn
    return $ nns!!50

online = flip (foldl' (flip (flip learn 0.002)))
postprocess :: Vector SNN.R -> Int
postprocess = fst . maximumBy cmp . zip [0..] . toList
  where cmp a b = compare (snd a) (snd b)

dotest :: NN -> IO ()
dotest nn = do
    testset <- uncurry zip <$> testData
    putStrLn "Test"
    let result = map (postprocess . flip forward nn . fst) testset
        expect = map (postprocess . snd) testset
        (co,wr)= partition (uncurry (==)) $ zip result expect
    putStrLn $ printf "correct: %d, wrong: %d" (length co) (length wr)
    -- putStrLn "a few errors are: "
    -- shuffle wr >>= printBox . hsep 2 left . map (text . show) . take 50
    -- putStrLn "and a few rights: "
    -- shuffle co >>= printBox . hsep 2 left . map (text . show) . take 50

{--
-- following piece of code was used during debug phase
--
dumpMain = do
    let nn = newNN [2,3,2]
        ds = [([0,0],[1,0]), ([0,1],[0,1]), ([1,0],[0,1]), ([1,1],[1,0])]
        inp = map (\(l1, l2) -> (fromList l1, fromList l2)) ds
        pass n0 = foldl' (flip (flip learn 0.5)) n0 inp
        ns = iterate pass nn
    dumpNN nn
    dumpNN (ns !! 1000)
    dumpNN (ns !! 2000)
    dumpNN (ns !! 3000)
    dumpNN (ns !! 4000)

stepMain = do
  let l1 = (2><2) [-0.0213,  0.0144, 0.0014,  -0.0001]
      b1 = fromList [0.0722, 0.0632]
      l2 = (2><2) [ -0.2354,  -0.2459, -0.2841,  -0.2772]
      b2 = fromList [0.5349, 0.5349]
      nt = [LL l1 b1, LL l2 b2]
      nn = NN { network = nt, activate = relu, activate' = relu' }
      step1 = learn (fromList [0,0], fromList [1,0]) 0.5
      step2 = learn (fromList [0,1], fromList [0,1]) 0.5
      step3 = learn (fromList [1,0], fromList [0,1]) 0.5
      step4 = learn (fromList [1,1], fromList [0,0]) 0.5
      nns = iterate step2 nn
  dumpNN nn
  dumpNN $ step4 $ step3 $ step2 $ step1 nn

dumpNN nn@(NN{network=net}) = do
    putStrLn $ render $ hsep 6 top [dump, apply]
  where
    dump = vcat left [text "=========== DUMP NN ===========", hsep 5 center1 (map prL net)]
    apply= vcat left [text "===========  APPLY  ===========", body]
    body = text $ show $ forward (fromList [1,0]) nn

prL l = hsep 1 center1 [prM (weights l), text "+", prB (biases l)]
prM m = vcat right (map prV (toRows m))
prV v = hsep 2 center1 (map prE (toList v))
prB b = vcat right (map prE (toList b))
prE f = text $ printf "%.4f" f

-- | Randomly shuffle a list
--   /O(N)/
shuffle :: [a] -> IO [a]
shuffle xs = withSystemRandom $ \gen -> do
        ar <- newArray n xs
        forM [1..n] $ \i -> do
            j  <- uniformR (i,n) gen
            vi <- readArray ar i
            vj <- readArray ar j
            writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (IOArray Int a)
    newArray n xs =  newListArray (1,n) xs

showLL :: LL -> Box
showLL (LL{weights=w,biases=b}) =
    let showColumn :: Vector Float -> Box
        showColumn v = vcat right $ map (\f -> text $ printf "%.4f" f) $ toList v
        wbox = hsep 1 center1 $ map showColumn $ toColumns w
        bbox = showColumn b
    in hsep 1 center1 [wbox, char '+', bbox]
--}

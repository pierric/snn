module Main where
import SNN
import Parser

main = do
    let nn = newNN [784, 30, 10]
    (labelset, imageset) <- trainingData
    return ()

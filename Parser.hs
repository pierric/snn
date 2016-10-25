module Parser where
import Data.Binary
import Data.Binary.Get
import qualified Data.ByteString.Lazy as BS
import Codec.Compression.GZip (decompress)
import Control.Monad

import Debug.Trace

type Pixel = Word8
type Image = [Pixel]
type Label = Word8

decodeImages :: Get [Image]
decodeImages = do
    mc <- getWord32be
    guard (mc == 0x00000803)
    [d1,d2,d3] <- many 3 getWord32be
    guard (d1 == 60000 && d2 == 28 && d3 == 28)
    many 60000 pic
  where
    pic = many (28*28) (get :: Get Pixel)

decodeLabels = do
    mc <- getWord32be
    guard (mc == 0x00000801)
    d1 <- getWord32be
    guard (d1 == 60000)
    many 60000 (get :: Get Label)

many :: Monad m => Int -> m a -> m [a]
many cnt dec = sequence (replicate cnt dec)

trainingData = do
    s <- decompress <$> BS.readFile "data/train-labels-idx1-ubyte.gz"
    t <- decompress <$> BS.readFile "data/train-images-idx3-ubyte.gz"
    return (runGet decodeLabels s, runGet decodeImages t)

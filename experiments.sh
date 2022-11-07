mkdir results
python3 -c "import newsroom; newsroom.main()" # On Newsroom
python3 -c "import realsumm; realsumm.main('abs')" # On RealSumm's abstractive subset
python3 -c "import realsumm; realsumm.main('ext')" # On RealSumm's extractive subset
python3 -c "import summeval; summeval.main()" # On SummEval

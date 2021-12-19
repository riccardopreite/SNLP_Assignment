To see how the command line works run:
python3 exercise_sheet4.py --help

optional arguments:
  -h, --help     show this help message and exit
  --line [LINE]  Index of line to observe > 0. Default 0
  --lr [LR]      learning rate. Default=0.01
  --test [TEST]  file of the model to load
  --mode [MODE]  model to test. "full" test on corpus_pos.txt, "example" test on corpus_example.txt, "obama" test on corpus_obama. Default full

An example run:
python3 exercise_sheet4.py --mode example

A full run:
python3 exercise_sheet4.py [--lr | --line]

A test run:
python3 exercise_sheet4.py --test <path/to/model>

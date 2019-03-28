### RUN ###
1. run main.py with the arg --opt.
2. insert a .json file. Notice that the extension is .json. An example .json file is provided.
3. The results will appear in 'result_path/name'.
4. running the same opt file without chaning the name of the results dir or moving it will add a timestamp to the name of the results dir.
5. The opt files in the supplementry are backward-compatible to the program.
6. in order to continue training a previous session, remove the '//' from the 'load' section and set the "dir" to the one containing the .pickle files of the features. Change the "level" to the level that was trained last.

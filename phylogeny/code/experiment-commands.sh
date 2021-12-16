#### IMPORTANT: original files must be processed through rewrite_nexus_files.py to ensure the taxa ordering matches ####
#### those generated for bootstrapping. Otherwise the trees generated in the trprobs files will not match           ####

SUFFIX_LIST=('-split-1-of-2' '-split-2-of-2')

for suffix in ${SUFFIX_LIST[@]}; do
    for model in JC HKY+C+G5 GTR+G+I mixed+G5; do
    	# Bayes
    	python code/create_mb_experiment_scc.py -m $model --by-codon-pos --hours 4  data/whale14sNuc${suffix}.nex hippopotamus --run --results new-results
    	# BayesBag
    	for a in .95 1; do
            python code/create_mb_experiment_scc.py -m $model --by-codon-pos -B 100 -a $a --hours 4  data/whale14sNuc${suffix}.nex hippopotamus --run --results new-results
	done
    done
done

for suffix in ${SUFFIX_LIST[@]}; do
   # Bayes
   python code/create_mb_experiment_scc.py -m mtmam+G5 --hours 4  data/whale14sAA${suffix}.nex hippopotamus --run --results new-results
   # BayesBag
   for a in .95 1; do
	python code/create_mb_experiment_scc.py -m mtmam+G5 -B 100 -a $a --hours 4  data/whale14sAA${suffix}.nex hippopotamus --run --results new-results
   done
done



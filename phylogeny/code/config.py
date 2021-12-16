

CONFIGS = {
'JC' : """lset nst=1;
   prset Statefreqpr=Fixed(0.25, 0.25, 0.25, 0.25);
   prset brlens=unconstrained: exp(10);""",

'HKY+C+G5' : """charset 1st_pos = 1-.\\3;
   charset 2nd_pos = 2-.\\3;
   charset 3rd_pos = 3-.\\3;
   partition by_codon = 3:1st_pos,2nd_pos,3rd_pos;
   set partition=by_codon;
   unlink Tratio = (all);
   unlink StateFreq = (all);
   unlink Ratemultiplier = (all);
   prset ratepr=variable;
   lset nst=2 rates=gamma Ngammacat=5;
   prset brlens=unconstrained: exp(10);""",

'GTR+G+I' : """charset 1st_pos = 1-.\\3;
      charset 2nd_pos = 2-.\\3;
      charset 3rd_pos = 3-.\\3;
      partition by_codon = 3:1st_pos,2nd_pos,3rd_pos;
      set partition=by_codon;
      unlink statefreq=(all) revmat=(all) shape=(all) pinvar=(all);
      prset ratepr=variable;
      lset nst=6 rates=invgamma;
      prset brlens=unconstrained: exp(10);""",

'mixed+G5' : """charset 1st_pos = 1-.\\3;
      charset 2nd_pos = 2-.\\3;
      charset 3rd_pos = 3-.\\3;
      partition by_codon = 3:1st_pos,2nd_pos,3rd_pos;
      set partition=by_codon;
      unlink statefreq=(all) revmat=(all) shape=(all) pinvar=(all);
      prset ratepr=variable;
      lset nst=mixed rates=gamma Ngammacat=5;
      prset brlens=unconstrained: exp(10);
      showmodel;""",

'mtmam+G5' : """lset rates=gamma Ngammacat=5;
      prset brlens=unconstrained: exp(10) Aamodelpr=Fixed(mtmam);
"""
}

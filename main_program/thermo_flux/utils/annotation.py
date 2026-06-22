from equilibrator_api import ComponentContribution, Q_
#cc = ComponentContribution()

def get_suitable_ids(met,cc, update_annotations = False ):
	found = False
	cpd = None
	formula = None
	inchi = None
	   
	for key, value in met.annotation.items():
		if not found:
			#print(key, value)
			cpd=cc.get_compound(str(key)+':'+value)
			if type(cpd) != type(None):
				found = True
				
	if not found:  
		try: 
			#cpd=cc.search_compound(met.name) #note this can return incorrect compounds and needs manual checking! 
			cpd = None
		except: 
			cpd = None

	
	annotation = {}

	if update_annotations:
		if found:
			for identifier in cpd.identifiers:
				namespace = identifier.registry.namespace
				if namespace not in met.annotation: #dont overwrite exisiting annotations
					if namespace not in annotation: #equilibrator can have multiple entries for identifiers - just take the first one 
						annotation[identifier.registry.namespace] = identifier.accession

			formula = cpd.formula
			inchi = cpd.inchi

	return cpd, annotation, formula, inchi
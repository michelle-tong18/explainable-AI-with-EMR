#Set up Elasticsearch client
import elasticsearch
from pprint import pprint as pp


class ImagePathQuery(object):
    def __init__(self,cohortlist,desc_weight,desc_plane,max_num_results=1e4):
        self.description_weight = self.check_description_weight(desc_weight)
        self.description_plane = self.check_description_plane(desc_plane)
        self.max_num_results = self.check_max_num_results(max_num_results)

        
        self.es = elasticsearch.Elasticsearch('http://localhost:9200') # Elasticsearch client - Accessible from the Wynton ic-app server  
        self.res = self.query_with_series_desc(cohortlist)  
        
    def check_description_weight(self,desc_weight):
        desc_weight_types = ['t1', 't2']
        if desc_weight not in desc_weight_types:
            raise ValueError("Invalid series descripting weighting type. Expected one of: %s" % desc_weight_types)
        return desc_weight                     
    
    def check_description_plane(self, desc_plane):
        desc_plane_types = ['sag', 'ax']
        if desc_plane not in desc_plane_types:
            raise ValueError("Invalid series descripting acquisition plane type. Expected one of: %s" % desc_plane_types)
        return desc_plane
    
    def check_max_num_results(self,max_num_results):
        if max_num_results>10000:
            raise ValueError("Maximum result window is 10000. Expected value <= 10,000")
        return max_num_results
        
    def query_with_series_desc(self,cohortlist):
        # For all the input accession numbers, find relevant images by matching the Series Description with specified weighting (t1, t2, pd, ...) and acquistiion plane (ax, sag, ...).
    
        #note - elasticsearch match is not case sensitive from what I can tell (Mar 2023)
        #note - size specifies the maxiumum number of results

        #search cohort and type of image
        cohort = [{'match': {'AccessionNumber': subject}} for subject in cohortlist]

        body = {
            'index': 'series',
            'query': {
                'bool': {
                    'should': cohort, # to "OR" match a list of PatienIDs, Accessions, or all UIDs, use 'should' instead of 'must' (these are keyword indexed)
                    'minimum_should_match': 1, # set's the "OR" match to 1 of the subject IDs
                    'must': [
                        #{'match': {'Modality': 'MR'}},
                        #{'match': {'StudyDescription': 'lumb'}},
                        #{'match': {'StudyDescription': 'spine'}},
                        {'match': {'SeriesDescription': self.description_weight}},
                        {'match': {'SeriesDescription': self.description_plane}},
                    ],
                },
            },
            'from': 0,
            'size': self.max_num_results,
        }

        res = self.es.search(**body)
        return res
    
    def update_query_with_filepaths(self,filepath_list):
        # For all input dcm folder filepaths
        
        #cohort = [{'multi_match': {'query': subject, 'fields':['FilePath']}} for subject in filepath_list]
        cohort = [{'match': {'FilePath': subject}} for subject in filepath_list]

        body = {
            'index': 'series',
            'query': {
                'bool': {
                    'should': cohort, # to "OR" match a list of PatienIDs, Accessions, or all UIDs, use 'should' instead of 'must' (these are keyword indexed)
                    'minimum_should_match': 1, # set's the "OR" match to 1 of the subject IDs
                },
            },
            'from': 0,
            'size': self.max_num_results,
        }

        res = self.es.search(**body)
        self.res = res
        return

    def get_total_num_results(self):
        return self.res['hits']['total']['value']
    
    def get_metadata(self, idx):
        return self.res['hits']['hits'][idx]['_source']
    
    def get_metadata_field(self,idx,col_name):
        try:
            return self.get_metadata(idx)[col_name]        
        except:
            return 'NaN'
    
    def get_all_metadata_field(self,col_name):
        #[series['_source'][col_name] for series in res['hits']['hits']]
        data_list = []
        for row in range(self.get_total_num_results()):
            data_list.append(self.get_metadata_field(row,col_name)) #handles missing data
        return data_list
 
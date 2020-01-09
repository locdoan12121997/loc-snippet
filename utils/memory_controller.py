'''
Use for face_reid application demo
'''
import numpy as np 
from sklearn.neighbors import NearestNeighbors

class MemoryController:
    def __init__(self, database, distance_thresh, new_id):
        # increse: assigned ID is more confidence but diffcult to assigned ID
        self.SEQUENCE_COUNT_THRES = 10
        self.database = database
        self.distance_thresh = distance_thresh
        # increase: feature is more stable, wrong ID is decreased, but the ability to assigning ID is reduced
        self.voting_T = 1.0
        self.group_database_threshold = 0.0
        # increase: features in galelery is bigger, voting is more confident, but searching is longer
        self.max_num_features = 500
        self.count_frame = None
        self.new_id = new_id
        self.max_id = new_id + 20

    def query_database(self, f, count_frame, is_blur):
        self.count_frame = count_frame
        self.is_blur = is_blur

        if len(self.database) == 0:
            person_id, idx = self.add_new_id(f)
            return [person_id, idx]
        # check if data only have 1 ID
        elif len(self.database) == 1:
            distances = self.compare_features(self.database[0]["features"], f)
            # find the index and min of distances to remove the feature
            idx_min = np.argmin(distances)
            person_id, idx = self.vote(distances=distances, id=0, current_feature=f, idx_min=idx_min, voting_threshold = self.voting_T)
            return [person_id, idx]
        # Check top 2 IDs closest to current features
        else:
            min_distances = []  # list store the min distance features in each ID
            idx_mins = []
            for p in self.database: # loop all ID in database
                distances = self.compare_features(p["features"], f)
                min_distances.append(np.min(distances))
                idx_mins.append(np.argmin(distances))
            
            min_distances = np.array(min_distances) 
            idx_mins = np.array(idx_mins)
            # take  top 2 closest ID
            top_2 = min_distances.argsort()[:2]

            if min_distances[top_2[0]] < self.distance_thresh and min_distances[top_2[1]] < self.distance_thresh:  
                centroid_0 = np.mean(self.database[top_2[0]]["features"], axis=0)
                centroid_1 = np.mean(self.database[top_2[1]]["features"], axis=0)
                if np.linalg.norm(centroid_0-centroid_1) <= self.group_database_threshold:
                    
                    if top_2[0]<top_2[1]:
                        person_id = self.database[top_2[0]]["id"]
                        idx = top_2[0]
                        for f_ in self.database[top_2[1]]['features']:
                            self.database[top_2[0]]['features'].append(f_)
                            self.database[top_2[0]]["count"] += 1
                            ###########################
                            if self.database[top_2[0]]["count"] == self.SEQUENCE_COUNT_THRES:
                                self.new_id = max([a["id"] for a in np.array(self.database)[:]]) + 1
                                del_idxs = []
                                for i in range(len(self.database)):
                                    if self.database[i]["count"] < self.SEQUENCE_COUNT_THRES:
                                        del_idxs.append((i))
                                self.database = np.delete(self.database, del_idxs)
                            ###########################
                    else:
                        person_id = self.database[top_2[1]]["id"]
                        idx = top_2[1]
                        for f_ in self.database[top_2[0]]['features']:
                            self.database[top_2[1]]['features'].append(f_)
                            self.database[top_2[1]]["count"] += 1
                            ###########################
                            if self.database[top_2[1]]["count"] == self.SEQUENCE_COUNT_THRES:
                                self.new_id = max([a["id"] for a in np.array(self.database)[:]]) + 1
                                del_idxs = []
                                for i in range(len(self.database)):
                                    if self.database[i]["count"] < self.SEQUENCE_COUNT_THRES:
                                        del_idxs.append((i))
                                self.database = np.delete(self.database, del_idxs)
                            ###########################
                else:                    
                    distances_id0 = self.compare_features(self.database[top_2[0]]["features"], f)
                    distances_id1 = self.compare_features(self.database[top_2[1]]["features"], f)
                    prob_close_features_0 = np.sum(distances_id0 <= self.distance_thresh) / np.shape(distances_id0)[0]
                    prob_close_features_1 = np.sum(distances_id1 <= self.distance_thresh) / np.shape(distances_id1)[0]
                    idx_min_id0 = np.argmin(distances_id0)
                    idx_min_id1 = np.argmin(distances_id1)
                    # 2 good
                    if prob_close_features_0 > self.voting_T and prob_close_features_1 > self.voting_T:      
                        norm_num_featuresid0 = np.shape(distances_id0)[0]/self.max_num_features  
                        norm_num_featuresid1 = np.shape(distances_id1)[0]/self.max_num_features          
                        f1_score_0 = prob_close_features_0 * np.shape(distances_id0)[0]
                        f1_score_1 = prob_close_features_1 * np.shape(distances_id1)[0]
                        #if (prob_close_features_0*np.shape(distances_id0)[0] > prob_close_features_1*np.shape(distances_id1)[0] ):
                        if f1_score_0>f1_score_1:
                            person_id = self.database[top_2[0]]["id"]
                            #idx = top_2[0]
                            idx = None
                        else:
                            person_id = self.database[top_2[1]]["id"]
                            #idx = top_2[1]
                            idx = None
                    # 1 good, 1 bad    
                    elif prob_close_features_0 > self.voting_T or prob_close_features_1 > self.voting_T:                         
                        if prob_close_features_0 > self.voting_T:
                            person_id, idx = self.add_new_features(top_2[0], f)
                            # remove outlier in other ID
                            #self.remove_features(top_2[1], idx_min_id1)
                        else:
                            person_id, idx = self.add_new_features(top_2[1], f)
                            # remove outlier in other ID
                            #self.remove_features(top_2[0], idx_min_id0)
                    else:# 2 bad
                        if self.new_id <= self.max_id:                        
                            person_id, idx = self.add_new_id(f)
                        else:
                            person_id, idx = None, None
                        #self.remove_features(top_2[0], idx_min_id0)
                        #self.remove_features(top_2[1], idx_min_id1)

            elif min_distances[top_2[0]] < self.distance_thresh or min_distances[top_2[1]] < self.distance_thresh:
                distances_id0 = self.compare_features(self.database[top_2[0]]["features"], f)
                distances_id1 = self.compare_features(self.database[top_2[1]]["features"], f)  
                idx_min_id0 = np.argmin(distances_id0)
                idx_min_id1 = np.argmin(distances_id1)

                if min_distances[top_2[0]] < self.distance_thresh:
                    goodID = top_2[0]
                    distances = distances_id0
                    idx_min = idx_min_id0
                else:
                    goodID = top_2[1]
                    distances = distances_id1
                    idx_min = idx_min_id1
           
                person_id, idx = self.vote(distances=distances, id=goodID, current_feature=f, idx_min=idx_min, voting_threshold=self.voting_T)
            else:
                if self.new_id <= self.max_id:
                    person_id, idx = self.add_new_id(f)
                else:
                    person_id, idx = None, None
            return [person_id, idx]

    def add_new_features(self, idx, f):  
        if len(self.database[idx]["features"]) <= self.max_num_features:
            if self.is_blur != True:
                self.database[idx]["features"].append(f)
        else:
            if self.is_blur != True:
                self.database[idx]["features"].append(f)
                self.database[idx]["features"] = self.database[idx]["features"][1:]

        
        if self.count_frame - self.database[idx]["frame"] <= 1:
            self.database[idx]["frame"] += 1
            self.database[idx]["count"] += 1
        else:
            if self.database[idx]["count"] < self.SEQUENCE_COUNT_THRES:
                self.database[idx]["count"] = 0 
            self.database[idx]["frame"] = self.count_frame
        real_id = self.database[idx]["id"]
        if self.database[idx]["count"] == self.SEQUENCE_COUNT_THRES:
            real_id = self.database[idx]["id"]
            self.new_id = max([a["id"] for a in np.array(self.database)[:]]) + 1
            del_idxs = []
            for i in range(len(self.database)):
                if self.database[i]["count"] < self.SEQUENCE_COUNT_THRES:
                    del_idxs.append((i))
            self.database = np.delete(self.database, del_idxs)
        return [real_id, idx]

    def add_new_id(self, f):
        data = {"id": self.new_id, "features": [f], "count": 0, "frame": self.count_frame}
        self.database = np.append(self.database, data)
        return [data["id"], self.new_id]

    def compare_features(self, memory_feats, f):
        '''
        return a numpy array, each elements is the distance between 
        feature in database with current feature
        '''
        return np.expand_dims(np.linalg.norm(memory_feats-f,axis=1),axis=0)

    def vote(self,distances, id, current_feature, idx_min, voting_threshold = 0.6):
        '''
        distance: list of distance between all feature in database with current feature [feature in database, 1]
        f:  current feature [feature embedding dimension]
        idx_min: scalar  , index of feature in database which has closest distance with current feature
        '''
        prob_is_same = np.sum(distances <= self.distance_thresh) / np.shape(distances)[0]
        if prob_is_same >= voting_threshold:  # Voting
            # Add New Feature to Existed ID 
            person_id, idx = self.add_new_features(id, current_feature)
        else:
            # Create New ID
            if self.new_id <= self.max_id:
                person_id, idx = self.add_new_id(current_feature)
            # Remove Outlier
            # del_idxs = np.where(distances <= self.distance_thresh)[0]
            # for del_idx in del_idxs:
            #     self.remove_features(id, del_idx)
            #     print('=======Remove database========')
        return [person_id, idx]

    def remove_features(self, idx, idx_feature):
        if len(self.database[idx]["features"]) > self.SEQUENCE_COUNT_THRES:
            del self.database[idx]["features"][idx_feature]
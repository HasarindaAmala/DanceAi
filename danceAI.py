

from vocal import VocalSync


def preProcess(ref_path,Test_path):

    #get both fps and make it equal
    #sync the sound and place test video on the ref video
    obj_vocal = VocalSync(ref_path,Test_path)
    obj_vocal.start_vocalSync()
    #trim and make both have same duration

    #return two videos(ref_sync,test_sync)


# =========================================================
# 8. Main Execution
# =========================================================
if __name__ == "__main__":
    preProcess("reference.mp4","test.mp4")
    #ref_sync,test_sync = preProcess(reference.mp4,test.mp4,)
    #ref_csv = genarate_csv(ref)
    #test_csv = genarate_csv(test)
    #similarities_csv = similarities(ref_csv,test_csv)
    #starting_time,end_time = draw_graph(similarities_csv)
    #score = genarate_marks(starting_time,end_time)
    #print(score)

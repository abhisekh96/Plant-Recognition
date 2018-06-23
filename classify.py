import tensorflow as tf
import sys
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


image_path = sys.argv[1]
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files/retrained_labels.txt")]
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    arr = {"predictions":{}}
    arr1 = []
    res = ""
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        arr['predictions'][human_string] = str(score)
        arr1.append(human_string)
    arr['result'] = arr1[0]
    confidence = int(round(float(arr["predictions"][arr1[0]])*100))
    arr['result_confidence'] = str(confidence) + "%"
    print(json.dumps(arr))

import argparse
import os
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects


def load_tf_graph(graph_path):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def _transform_ops():
    return [
        'add_default_attributes',
        'strip_unused_nodes',
        'remove_nodes(op=Identity, op=CheckNumrics)',
        'fold_batch_norms',
        'fold_old_batch_norms',
        'sort_by_execution_order',
        'round_weights(num_steps=1024)',
        'fold_constants(ignore_errors=true)',
        'merge_duplicate_nodes',
        'remove_control_dependencies',
        'merge_duplicate_nodes'
    ]


def ModelFreezer(model_fname, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if model_fname[-2:] == 'h5':
        tf.keras.backend.set_learning_phase(0) 

        # need to define model if you have custom_object. model = get_model() then model.load_weights(H5_FILE). load_model just work if u don't have any custom_object
        model = load_model(model_fname, compile=False, custom_objects={"swish": tf.nn.relu})
        session = tf.keras.backend.get_session()
        graph = session.graph.as_graph_def()
    else:
        graph = load_tf_graph(model_fname)

    transformed_graph_def = TransformGraph(
        graph, 
        inputs=[model.input.name[:-2]],
        outputs=[model.output.name[:-2]],
        transforms=_transform_ops()
    )

    const_graph_def = graph_util.convert_variables_to_constants(
        session,
        transformed_graph_def,
        [model.output.name[:-2]]
    )

    try:
        optimize_for_inference_lib.ensure_graph_is_valid(const_graph_def)
        tf.train.write_graph(const_graph_def, output_dir, 'optimized_frozen.pb', as_text=False)
    except ValueError as e:
        print('Graph is invalid - {}'.format(e))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_file', help='Path to trained model')
    parser.add_argument('--pb_file', help='Path to trained model')
    parser.add_argument('--output_dir', default='./', help='Path to save the frozen graphs')

    args = parser.parse_args()
        
    try:
        checkpoint_path = tf.train.get_checkpoint_state(args.tacotron_checkpoint).model_checkpoint_path
        print('loaded model checkpoint at {}'.format(checkpoint_path))
    except:
        print('error')
    
    ModelFreezer(args.h5_file, args.output_dir)

if __name__ == '__main__':
    main()


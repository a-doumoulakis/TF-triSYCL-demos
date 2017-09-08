import tensorflow as tf

def testOclOp(self):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    conf = tf.ConfigProto(allow_soft_placement=False,
                          device_count={'SYCL': 3})
    sess = tf.InteractiveSession(config=conf)
    with tf.device('/cpu:0'):
        arg1 = tf.fill([6,3,2], 11.5, name="arg1")
        arg2 = tf.fill([6,3,2], 10.5, name="arg2")
        arg3 = tf.fill([6,3,2], 5.0, name="arg3")
        arg4 = tf.fill([6,3,2], 2.0, name="arg4")
        arg5 = tf.fill([6,3,2], 2.0, name="arg5")

    with tf.device('/device:SYCL:0'):
        add_node = arg1 + arg2

    with tf.device('/device:SYCL:2'):
        mul_node = tf.user_ops.ocl_native_op(input_list=[arg3, arg4],
                                           output_type=tf.float32,
                                           shape=[6,3,2],
                                           file_name="/home/anastasi/OpenCLKernels/VecMul.cl",
                                           kernel_name="vector_mul",
                                           is_binary=False)
    with tf.device('/device:SYCL:1'):
        result = tf.user_ops.ocl_native_op(input_list=[add_node, mul_node, arg5],
                                           output_type=tf.float32,
                                           shape=[6,3,2],
                                           file_name="/home/anastasi/OpenCLKernels/VecAddMul.xclbin",
                                           kernel_name="vector_add_mul",
                                           is_binary=True)


    res = sess.run([result])
    print(res[0])
    writer = tf.summary.FileWriter('/tmp/tensorflow/logs/test',
                                   sess.graph)
    writer.close()


if __name__ == "__main__":
    tf.app.run(main=testOclOp)

'''
    Image Classification TFDS Batch Player Block's source code
    Auther: Ahmed Hendawy - YonoHub Developer Advocate
    Date: 01.06.2020
'''

# Tensorflow Datasets
import tensorflow_datasets as tfds
# Vision
import cv2
# Utils
import time
# Yonoarc Utils
from yonoarc_utils.image import from_ndarray
from yonoarc_utils.header import set_timestamp
# Messages
from perception_msgs.msg import ImageBatch, LabelArray, Label
from std_msgs.msg import Header

class image_classification_tfds_batch_player:
    dataset_list=["beans","caltech101","caltech_birds2010","caltech_birds2011",
                "cars196","cassava","cats_vs_dogs","cifar10","cifar100","cifar10_corrupted",
                "citrus_leaves","cmaterdb","colorectal_histology","colorectal_histology_large","cycle_gan","deep_weeds",
                "dmlab","dtd","emnist","eurosat","fashion_mnist","food101","geirhos_conflict_stimuli","horses_or_humans",
                "i_naturalist2017","imagenet_resized","imagenette","imagewang","kmnist","lfw","malaria","mnist","mnist_corrupted",
                "omniglot","oxford_flowers102","oxford_iiit_pet","patch_camelyon","places365_small","plant_leaves",
                "plant_village","plantae_k","quickdraw_bitmap","rock_paper_scissors","smallnorb","so2sat","stanford_dogs",
                "stanford_online_products","stl10","sun397","svhn_cropped","tf_flowers","uc_merced","visual_domain_decathlon"]
    
    def on_start(self):
        self.dataset=self.dataset_list[self.get_property("dataset")]
        self.batch_size=self.get_property("batch_size")
        self.split=self.get_property("split")
        self.loop=self.get_property("loop")
        self.shuffle=self.get_property("shuffle")
        self.play=False
        # Build the `tf.data.Dataset` pipeline.
        try:
            self.alert("Downloading the %s dataset"%self.dataset,"INFO")
            self.ds, self.info=tfds.load(self.dataset, split=self.split, shuffle_files=self.shuffle, with_info=True)
            if self.loop:
                self.ds = self.ds.repeat().shuffle(self.info.splits[self.split].num_examples,seed=0).batch(self.batch_size)
            else:
                self.ds = self.ds.shuffle(self.info.splits[self.split].num_examples).batch(self.batch_size)
            self.alert("%s dataset is downloaded successfully"%self.dataset,"INFO")        
        except ValueError as error:
            self.alert("This split you chose is not available. Please check the available splits at https://www.tensorflow.org/datasets/catalog/overview","ERROR")


    def run(self):
        while True:
            if self.play==True:
                self.play=False
                # `tfds.as_numpy` converts `tf.Tensor` -> `np.array`
                for ex in tfds.as_numpy(self.ds):
                    start=time.time()
                    header=Header()
                    set_timestamp(header,time.time())
                    image_batch_msg=ImageBatch()
                    image_batch_msg.header=header
                    if not ex['image'].shape[3]==1:
                        image_batch_msg.Batch=[from_ndarray(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),header) for img in ex["image"]]
                    else:
                        image_batch_msg.Batch=[from_ndarray(img,header) for img in ex["image"]]
                    
                    labels_msg=LabelArray()
                    labels_msg.header=header
                    for label in ex['label']:
                        label_msg=Label()
                        label_msg.header=header
                        label_msg.confidence=1.0
                        # `int2str` returns the human readable label ('dog', 'car',...)
                        label_msg.class_name=self.info.features['label'].int2str(label)
                        labels_msg.labels.append(label_msg)
                    

                    self.publish("image_batches",image_batch_msg)
                    self.publish("labels",labels_msg)
                    end=time.time()-start
                    if (1.0/self.get_property("frame_rate")) > end:
                        time.sleep((1.0/self.get_property("frame_rate"))-end)             

            time.sleep(0.001)
            
    def on_button_clicked(self, button_key):
        self.play=True

                



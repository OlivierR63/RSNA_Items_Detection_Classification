import tensorflow as tf

full_path_name = r"C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification\data\lumbar_spine\tfrecords\100206310.tfrecord"

raw_dataset = tf.data.TFRecordDataset(full_path_name)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    # Extraction brute de la feature
    meta_bytes = example.features.feature['metadata'].bytes_list.value[0]
    print(f"HEXADECIMAL REEL : {meta_bytes.hex(' ')}")
    print(f"LONGUEUR TOTALE : {len(meta_bytes)} octets")
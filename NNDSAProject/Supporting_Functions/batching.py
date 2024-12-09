def batch_split(features_array,output_array,batch_size):
    batches_features=[features_array[i:i+batch_size] for i in range(0,len(features_array),batch_size)]
    batches_output=[output_array[i:i+batch_size] for i in range(0,len(output_array),batch_size)]
    return batches_features,batches_output

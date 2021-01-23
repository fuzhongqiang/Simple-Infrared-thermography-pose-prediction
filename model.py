from tensorflow import keras

def build_model():
    inputs = keras.Input(shape=(8,23,27,1))

    x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
    # x = keras.layers.BatchNormalization()(x)  
    x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
    x_c1 = keras.layers.BatchNormalization()(x)

    x_c2 = keras.layers.Conv3DTranspose(64,(1,3,3),strides=(1, 2,2), padding='same')(x_c1)
    x_c2 = keras.layers.BatchNormalization()(x_c2)
    x_c2 = keras.layers.UpSampling3D(size=(1,2,2))(x_c1) + x_c2

    x_c2 = keras.layers.Conv3D(64,(1,3,3),padding='same',activation='relu')(x_c2)
    x_c2 = keras.layers.BatchNormalization()(x_c2)

    x_c3 = keras.layers.Conv3DTranspose(64,(1,3,3),strides=(1, 2,2), padding='same')(x_c2)
    x_c3 = keras.layers.BatchNormalization()(x_c3)
    x_c3 = keras.layers.UpSampling3D(size=(1,2,2))(x_c2) + x_c3

    x_c3 = keras.layers.Conv3D(64,(1,3,3),padding='same',activation='relu')(x_c3)
    x_c3 = keras.layers.BatchNormalization()(x_c3)

    x_c4 = keras.layers.Conv3DTranspose(64,(1,3,3),strides=(1, 2,2), padding='same')(x_c3)
    x_c4 = keras.layers.BatchNormalization()(x_c4)
    x_c4 = keras.layers.UpSampling3D(size=(1,2,2))(x_c3) + x_c4

    x_c4 = keras.layers.Conv3D(128,(1,3,3),padding='same',activation='relu')(x_c4)

    outputs = keras.layers.Conv3D(1,(1,3,3), padding='same')(x_c4)
    return keras.Model(inputs,outputs)
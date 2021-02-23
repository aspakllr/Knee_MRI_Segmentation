from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Dropout, UpSampling3D


def unet_model(img_depth, img_rows, img_cols, num_of_labels ):

    inputs = Input(shape=(img_depth, img_rows, img_cols, 1))  # input shape (None, 64, 128, 160, 1)

    # Encoder
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    drop1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(drop1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    drop2 = Dropout(0.1)(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(drop2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    drop3 = Dropout(0.1)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(drop3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    drop4 = Dropout(0.1)(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(drop4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    drop5 = Dropout(0.1)(conv5)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(drop5)


    # Decoder
    up6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 1))(conv5))
    merge6 = concatenate([conv4, up6], axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(merge6)
    drop6 = Dropout(0.1)(conv6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(drop6)

    up7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge7)
    drop7 = Dropout(0.1)(conv7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(drop7)

    up8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge8)
    drop8 = Dropout(0.1)(conv8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(drop8)

    up9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge9)
    drop9 = Dropout(0.1)(conv9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(drop9)

    output = Conv3D(num_of_labels, (1, 1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[output])
    # model.summary()
    # plot_model(model, to_file='unetModel.png', show_shapes=True)

    return model


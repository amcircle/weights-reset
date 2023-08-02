from matplotlib import pyplot as plt
import time

def current_milli_time_str():
    return str(round(time.time() * 1000))

def plot_history(h):
    plt.plot(h['accuracy'])
    plt.plot(h['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def get_dataset_name(ds):
    return ''.join(ds.value.split('/'))

def get_csv_filename(title, ds_name):
    t = current_milli_time_str()
    return f'{title}_{ds_name}_{t}.csv'
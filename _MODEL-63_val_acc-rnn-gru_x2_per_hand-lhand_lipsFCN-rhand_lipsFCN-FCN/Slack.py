from slack_sdk import WebClient
import matplotlib.pyplot as plt
# from io import BytesIO #to post matplotlib graphs to slack
import tensorflow as tf

class SlackCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        token=None,
        channel='C04SMG639GC',
        message: str = None,
        start_message: str = '',
    ):
        self.start_message = start_message
        self.token = token
        self.channel = channel
        self.logs = {}
        if message is None:
            self.message = (
                'Epoch {epoch:03d}:\n'
                + 'loss:{loss:.4f} val_loss:{val_loss:.4f}\n'
                + 'acc:{accuracy:.4f} val_acc:{val_accuracy:.4f}\n'
            )
        else:
            self.message = message
        self.client = WebClient(token=self.token)
        

    def on_train_begin(self, logs=None):
        self.send_message(
            f'Starting new model...\n{self.start_message}'
        )

    def on_epoch_end(self, epoch, logs={}):
        # Share message of model in thread
        self.send_message(
            self.message.format(epoch=epoch+1,**logs),
            thread_ts=self.ts,
            reply_broadcast=False,
        )
        # Keep track of the metrics via log
        for metric, val in logs.items():
            prev = self.logs.get(metric, [])
            prev.append(val)
            self.logs[metric] = prev

    def on_train_end(self, logs):
        # Create a images
        fig, (ax_acc, ax_loss) = plt.subplots(ncols=2, figsize=(16,8))

        ax_acc.plot(self.logs['accuracy'])
        ax_acc.plot(self.logs['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        ax_loss.plot(self.logs['loss'])
        ax_loss.plot(self.logs['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        fname = 'temp-loss_acc_curves.png'
        fig.savefig(fname, format='png')
        # Bot uploads image for itself
        image = self.client.files_upload(
            channel=self.channel,
            initial_comment='Loss & Acc curve from recent model run',
            file=fname,
        )
        file_url = image['file']['permalink']
        # In thread, final message with image link
        self.send_message(
            f'training done!! {file_url}',
            thread_ts=self.ts,
            reply_broadcast=False,
        )
        plt.show()
        
    def send_message(self, text: str, **kwargs):
        try:
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=text,
                **kwargs,
            )
            self.ts = response.data['ts']
        except:
            pass

from utils import compute_loss_accuracy

def test_and_save(net, dataloader, device, writer=None, data_name=None, epoch=None):
    loss, acc = compute_loss_accuracy(net, dataloader, device)
    print(data_name, 'loss_p: {:.4f} acc_p: {:.4f}'.format(loss[0], acc[0]))
    print(data_name, 'loss_c: {:.4f} acc_c: {:.4f}'.format(loss[1], acc[1]))
    print(data_name, 'loss_w: {:.4f} acc_w: {:.4f}'.format(loss[2], acc[2]))
    if writer is not None:
        writer.add_scalar('ACC/' + data_name +'_p', acc[0], epoch)
        writer.add_scalar('ACC/' + data_name +'_p', acc[1], epoch)
        writer.add_scalar('ACC/' + data_name +'_p', acc[2], epoch)
    return loss, acc

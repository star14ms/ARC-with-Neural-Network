import torch
from torch import nn
from tqdm import tqdm
import sys
sys.path.append('./src/')

from arc.constants import get_challenges_solutions_filepath
from classify import ARCDataClassifier
from data import ARCDataset
from arc.model.substitute.v2_CL_encode import PixelEachSubstitutor
from arc.utils.visualize import visualize_image_using_emoji
from arc.preprocess import one_hot_encode


def train(xs_train, ts_train, xs_test, ts_test, model, alpha, gamma, n_iteration=50, max_recursion=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    acc_max_list = [(x.argmax(dim=1) == t.argmax(dim=1)).sum().item() / (x.size(1) * x.size(2)) for x, t in zip(xs_train, ts_train)]
    n_iters_perfect = 0

    for e in (pbar:= tqdm(range(n_iteration))):
        n_task_correct = 0

        for i, (x, t) in enumerate(zip(xs_train, ts_train)):
            x = x.unsqueeze(0)
            t = t.unsqueeze(0)
            y_prev = x.clone()
            
            acc_inital = (x.argmax(dim=1) == t.argmax(dim=1)).sum().item() / (x_size:= x.size(2) * x.size(3))
            acc_max = acc_inital

            for depth in range(max_recursion):
                if depth != 0:
                    y_prev = y
                    max_indices = torch.argmax(y_prev, dim=1)
                    y_prev = torch.scatter(torch.zeros_like(y_prev), 1, max_indices.unsqueeze(1), 1)
                
                y = model(y_prev) # [N, C, H, W]

                n_pixel_correct = (y.argmax(dim=1) == t.argmax(dim=1)).sum().item()
                acc = n_pixel_correct / x_size
                # is_new_correct_pixel = torch.any(correct.int() - correct_inital.int() == 1)

                if acc > acc_max_list[i]:
                    acc_max_list[i] = acc
                    y_changed = torch.where(y.argmax(dim=1) != y_prev.argmax(dim=1), 1, 0) # (N, H, W)
                    y_changed_green = torch.where(y_changed == 1, 3, 0)
                    visualize_image_using_emoji(x[0], t[0], y[0], one_hot_encode(y_changed_green[0]), titles=['Input', 'Target', 'Output', 'Changed'])
                
                if acc < acc_max or n_pixel_correct == x_size or torch.all(y_prev.argmax(dim=1) == y.argmax(dim=1)):
                    break

                if acc > acc_max:
                    acc_max = acc
                
            loss = loss_fn(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_pixel_correct == x_size:
                n_task_correct += 1

        pbar.set_description(f"Loss: {loss.item():.4f}, Recursion: {depth+1}, Correct Task: {n_task_correct}/{len(xs_train)}")

        if n_task_correct == len(xs_train):
            test(model, xs_test, ts_test, max_recursion)
            print(n_iters_perfect, e+1)
            n_iters_perfect += 1
        else:
            n_iters_perfect = 0

        if n_iters_perfect == 5:
            break

    return model


def test(model, xs_test, ts_test, max_recursion=30, show_depth=False):
    n_correct_tasks = 0
    for x, t in zip(xs_test, ts_test):
        x = x.unsqueeze(0)
        t = t.unsqueeze(0)
        y_prev = x.clone()
        
        ys = []

        for depth in range(max_recursion):
            if depth != 0:
                y_prev = y
                max_indices = torch.argmax(y_prev, dim=1)
                y_prev = torch.scatter(torch.zeros_like(y_prev), 1, max_indices.unsqueeze(1), 1)
            
            y = model(y_prev)
            ys.append((y, 'Depth {}'.format(depth+1)))
            c_decoded = torch.where(y.argmax(dim=1) == t.argmax(dim=1), 3, 2)

        str_visualization = ''
        if len(ys) > 1 and show_depth:
            xytc = [(x, 'Input')] + ys + [(t, 'Target')] + [(c_decoded, 'Correct')]
            xytc_batches = [xytc[i:i+4] for i in range(0, len(xytc), 4)]
            for xytc_batch in xytc_batches:
                titles = [title for _, title in xytc_batch]
                xytcs = [xytc for xytc, _ in xytc_batch]
                str_visualization += visualize_image_using_emoji(*xytcs, titles=titles, return_str=True)
        else:
            str_visualization = visualize_image_using_emoji(x, y, t, c_decoded, return_str=True)
        print(str_visualization)

        if torch.all(y.argmax(dim=1) == t.argmax(dim=1)):
            n_correct_tasks += 1
            
    print(f"Correct Tasks: {n_correct_tasks}/{len(xs_test)}")
        

if __name__ == '__main__':
    alpha = 0.1 # Learning Rate
    gamma = 0.9 # Discount Factor
    n_iteration = 300
    n_recursion = 30

    data_category = 'train'
    task_id = '3bd67248'
    challenges, solutions = get_challenges_solutions_filepath(data_category)

    filter_funcs = (ARCDataClassifier.in_data_codes_f([task_id], reorder=True),)
    dataset = ARCDataset(challenges, solutions, one_hot=True, augment_data=False, augment_test_data=False, filter_funcs=filter_funcs)

    torch.set_printoptions(precision=2, sci_mode=False)
    xs_train, ts_train, xs_test, ts_test, task_id = dataset[0]

    model = PixelEachSubstitutor(
        n_range_search=1, W_kernel_max=3, H_kernel_max=3, 
        emerge_color=True, vec_abs=False,
        C_dims_encoded=[10, 3], L_dims_encoded=[9, 4], L_dims_decoded=[9, 1],
        # pad_class_initial=-1
    )

    model = train(xs_train, ts_train, xs_test, ts_test, model, alpha, gamma, n_iteration)
    test(model, xs_test, ts_test, n_recursion)

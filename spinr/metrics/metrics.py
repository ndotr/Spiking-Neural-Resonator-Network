import numpy as np
import scipy as sp
import cupy as cp
import cupyx.scipy.signal
# Just for debugging
import matplotlib.pyplot as plt


import logging
log = logging.getLogger()

#TODO:
# matching metric
# distance/angle error metric
# mutlitarget metric: count save matches etc.

def is_peak(data, peak, threshold):
    return data[peak[0], peak[1]] > threshold

def detect_peak_in_target_window(data, target_maps, window_size):

    r_offset, a_offset = window_size

    n_frames, n_ranges, n_angles = target_maps.shape
    n_frames, n_ranges, n_angles = data.shape

    peaks = []

    for f in range(n_frames):
        r_idxs, a_idxs = np.where(target_maps[f]>0)
        for r_idx, a_idx in zip(r_idxs, a_idxs):
            r_start = r_idx - r_offset - 1
            r_end = r_idx + r_offset
            if r_start < 0:
                r_start = 0
            if r_end > n_ranges-1:
                r_end = n_ranges-1
            a_start = a_idx - a_offset - 1
            a_end = a_idx + a_offset
            if a_start < 0:
                a_start = 0
            if a_end > n_angles-1:
                a_end = n_angles-1

        data_window = data[f, r_start:r_end, a_start:a_end]

        peak = np.unravel_index(np.argmax(data_window, axis=None), data_window.shape)
        r_peak = peak[0] + r_start
        a_peak = peak[1] + a_start
        peaks.append([r_peak, a_peak])

    return peaks

def get_window_limits(idx, offset, max_idx):

    idx_start = idx - offset
    idx_end = idx + offset + 1
    if idx_start < 0:
        idx_start = 0
    if idx_end > max_idx-1:
        idx_end = max_idx-1

    return idx_start, idx_end

def eval_peaks_in_target_window(data, targets, target_maps, window_size, distance_range, angle_range,
                                max_flag=False):

    d_offset, a_offset = window_size

    n_frames, n_distances, n_angles = target_maps.shape
    n_frames, n_distances, n_angles = data.shape

    peak_evas = []
    target_windows = []
    mean_snr = 0
    mean_ssr = 0
    mean_ssr_window = 0
    mean_d_error = 0
    mean_a_error = 0
    mean_signal = 0
    mean_noise = 0
    mean_sum = 0
    n_peaks = 0

    sum=0

    for f in range(n_frames):
        peak_evas.append([])
        target_windows.append([])
        d_idxs, a_idxs = np.where(target_maps[f]>0)
        for d_idx, a_idx in zip(d_idxs, a_idxs):
            # add target to last frame
            peak_evas[-1].append([])
            d_start, d_end = get_window_limits(d_idx, offset=d_offset, max_idx=n_distances)
            a_start, a_end = get_window_limits(a_idx, offset=a_offset, max_idx=n_angles)

            data_window = data[f, d_start:d_end, a_start:a_end]
            target_windows[-1] = [a_start, a_end, d_start, d_end]
            #target_window = target_maps[f, d_start:d_end, a_start:a_end]

            #peaks = np.where(data_window>0)
            #peaks = np.where((data_window==np.max(data_window)))
            peaks = np.where((data_window==np.max(data_window)) & (data_window>0))
            window_sum = np.sum(data_window)

            if len(peaks[0])>0:
                for p0, p1 in zip(peaks[0][0:1], peaks[1][0:1]):
                    d_peak_idx = p0 + d_start
                    a_peak_idx = p1 + a_start
                    # metrics

                    # SNR
                    signal  = data_window[p0, p1]
                    #signal  = data[f, r_peak_idx, a_peak_idx]
                    sum = np.sum(data[f])
                    noise = sum - signal
                    ssr = signal**2/sum**2
                    snr = signal**2/noise**2
                    ssr_window = signal**2/window_sum**2

                    # distance error
                    pos = targets[0][f][1][0]
                    d = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                    a = np.arctan(pos[1]/pos[0])
                    d_error = d - distance_range[d_peak_idx]
                    # angle error
                    a_error = a - angle_range[a_peak_idx]

                    mean_snr += snr
                    mean_ssr += ssr
                    mean_ssr_window += ssr_window
                    mean_signal += np.abs(signal)
                    mean_noise += np.abs(noise)
                    mean_d_error += np.abs(d_error)
                    mean_a_error += np.abs(a_error)
                    mean_sum += np.abs(sum)
                    n_peaks += 1

                    peak_evas[-1][-1].append({'idx_d': int(d_peak_idx),
                                              'idx_a': int(a_peak_idx),
                                              'signal': float(signal),
                                              'noise': float(noise),
                                              'sum': float(sum),
                                              'snr': float(snr),
                                              'ssr': float(ssr),
                                              'ssr window': float(ssr_window),
                                              'distance error': float(d_error),
                                              'angle error': float(a_error/np.pi*180),
                                              })
            else:
                sum = np.sum(data[f])
                peak_evas[-1][-1].append({'idx_d': -1,
                                        'idx_a': -1,
                                        'signal': 0,
                                        'noise': float(sum),
                                        'sum': float(sum),
                                        'snr': 0,
                                        'ssr': 0,
                                        'ssr window': 0,
                                        'distance error': -1,
                                        'angle error': -1,
                                        })



    if n_peaks>0:
        eva = {'signal': float(mean_signal/n_peaks),
                'noise': float(mean_noise/n_peaks),
                'sum': float(mean_sum/n_peaks),
                'snr': float(mean_snr/n_peaks),
                'ssr_peaks': float(mean_ssr/n_peaks),
                'ssr_targets': float(mean_ssr/n_frames),
                'ssr_window_targets': float(mean_ssr_window/n_frames),
                'ssr_window_peaks': float(mean_ssr_window/n_peaks),
                'distance error': float(mean_d_error/n_peaks),
                'angle error': float(mean_a_error/n_peaks),
                'num peaks': int(n_peaks),
                'detection rate': float(n_peaks/n_frames)}
    else:
        eva = {'signal': 0,
                'noise': int(sum),
                'sum': int(sum),
                'snr': 0,
                'ssr_targets': 0,
                'ssr_peaks': 0,
                'ssr_window_targets': 0,
                'ssr_window_peaks': 0,
                'distance error': -1,
                'angle error': -1,
                'num peaks': int(n_peaks),
                'detection rate': float(n_peaks/n_frames)}

    return eva, peak_evas, target_windows


def ssr(data, target_maps, kernel, tags):

    if kernel is not None :
        for i in range(target_maps.shape[0]):
            target_maps[i] = sp.signal.convolve2d(target_maps[i], kernel, mode='same')

    sums = np.sum(data, axis=(1,2))
    signals = np.sum(np.einsum('ijk,ijk-> ijk', data, target_maps), axis=(1,2))#/np.sum(target_maps, axis=(1,2))
    ssrs = signals/sums

    mask = (sums > 0)
    signal = signals[mask]
    sum = sums[mask]

    ssr = np.sum(signal/sum/np.sum(target_maps[0]))/np.sum(mask)
    ssr2 = np.sum(signals)/np.sum(sums)
    detection_rate = np.sum(ssrs/np.sum(target_maps[0])>1)/len(ssrs)

    ret = np.array([tags, ssr, ssr2])
    #ret = [ssr, ssr2, ssrs]
    #for t in reversed(tags):
    #    ret.insert(0,t)

    return ret

def ssr_gpu(data, target_maps, kernel, tags):

    #data = cp.array(data)
    if kernel is not None :
        for i in range(target_maps.shape[0]):
            target_maps[i] = cupyx.scipy.signal.convolve2d(target_maps[i], kernel, mode='same')

    sums = cp.sum(data, axis=(1,2))
    signals = cp.sum(np.einsum('ijk,ijk-> ijk', data, target_maps), axis=(1,2))#/np.sum(target_maps, axis=(1,2))
    ssrs = signals/sums

    mask = (sums > 0)
    signal = signals[mask]
    sum = sums[mask]

    ssr = cp.sum(signal/((sum*cp.sum(target_maps[0]))+1e-8))/(cp.sum(mask)+1e-8)
    ssr2 = cp.sum(signals)/cp.sum(sums)
    detection_rate = cp.sum(ssrs/cp.sum(target_maps[0])>1)/len(ssrs)

    ret = np.array([tags, ssr.get(), ssr2.get()])
    #for t in reversed(tags):
    #    ret.insert(0,t)

    return ret

def ssr_pertarget_gpu(data, target_maps, kernel, tags):

    target_maps = target_maps[...,:200,:]
    data = data[...,:200,:]

    #data = cp.array(data)
    ret = np.zeros((target_maps.shape[0], 4))
    if kernel is not None:
        for target_idx in range(target_maps.shape[0]):
            for i in range(target_maps.shape[1]):
                target_maps[target_idx, i] = cupyx.scipy.signal.convolve2d(target_maps[target_idx, i], kernel, mode='same')

    sums = cp.repeat(cp.expand_dims(cp.sum(data, axis=(1,2)), axis=0), target_maps.shape[0], axis=0)
    signals = cp.sum(cp.einsum('ijk, tijk-> tijk', data, target_maps), axis=(2,3))#/np.sum(target_maps, axis=(1,2))
    #ssrs = signals/sums

    #mask = (sums > 0)
    #signal = signals[mask]
    #sum = sums[mask]
    ssr = cp.sum(signals/(sums+1e-8)/(cp.sum(target_maps, axis=(2,3))+1e-8), axis=1)/target_maps.shape[1]
    ssr2 = cp.sum(signals, axis=1)/cp.sum(sums, axis=1)
    #detection_rate = cp.sum(ssrs/cp.sum(target_maps[0])>1)/len(ssrs)
    for target_idx in range(target_maps.shape[0]):
        ret[target_idx,:] = np.array([tags, target_idx, ssr[target_idx].get(), ssr2[target_idx].get()])
    #for t in reversed(tags):
    #    ret.insert(0,t)

    return ret

def cfar_detection(data, target_maps, kernel, factors, offsets, hard_threshold=None, detection_area=None, normalize=False, tags='Tag'):

    #precision_list = []
    #recall_list = []
    #ext_precision_list = []
    #ext_recall_list = []
    #fp_list = []
    #fn_list = []
    #iou_list = []
    #niou_list = []

    rows = []
    rows = np.zeros((len(factors)*len(offsets), 7))

    for m, data_offset in enumerate(offsets):
        
        #precision_list.append([])
        #recall_list.append([])
        #ext_precision_list.append([])
        #ext_recall_list.append([])
        #fp_list.append([])
        #fn_list.append([])
        #iou_list.append([])
        #niou_list.append([])


        proc_data = data + data_offset
        if normalize:
            proc_data = np.einsum('ijk,i->ijk', proc_data, 1/np.max(data,axis=(1,2)))
    
        # CA - CFAR
        cfar_thresh_maps = np.zeros_like(target_maps)
        for i in range(data.shape[0]):
            cfar_thresh_maps[i] = (sp.signal.convolve2d(np.pad(proc_data[i], 
                                                               pad_width=((kernel.shape[0], kernel.shape[0]), (kernel.shape[1], kernel.shape[1])),
                                                                mode='symmetric'), 
                                                        kernel, mode='same')/np.sum(kernel))[kernel.shape[0]:-kernel.shape[0],kernel.shape[1]:-kernel.shape[1]]

        if detection_area is not None:
            extended_target_maps = np.zeros_like(target_maps)
            for i in range(target_maps.shape[0]):
                extended_target_maps[i] = np.clip(sp.signal.convolve2d(target_maps[i], detection_area, mode='same'), a_min=None, a_max=1)
        else:
            extended_target_maps = target_maps

        for n, factor in enumerate(factors):
            cfar_maps = cfar_thresh_maps*factor
            peak_maps = data > (cfar_maps)

            if detection_area is not None:
                extended_peak_maps = np.zeros_like(peak_maps)
                for i in range(target_maps.shape[0]):
                    extended_peak_maps[i] = sp.signal.convolve2d(peak_maps[i], detection_area, mode='same')>0
            else:
                extended_peak_maps = peak_maps

            # Detection rate and false detection rate
            tp_maps = (peak_maps * target_maps) > 0
            fp_maps = (peak_maps - target_maps) > 0 # peak is there, but no target
            fn_maps = (target_maps - peak_maps) > 0 # target is there, but no peak

            intersection_maps = (extended_peak_maps * extended_target_maps) > 0
            union_maps = (extended_peak_maps + extended_target_maps) > 0
            nintersection_maps = (extended_peak_maps - extended_target_maps) > 0

            recall = np.sum(np.sum(tp_maps,axis=(1,2))/np.sum(target_maps, axis=(1,2)))/target_maps.shape[0]
            #recall_list[-1].append(recall)

            fp = np.sum(np.sum(fp_maps, axis=(1,2))/np.clip(np.sum(peak_maps,axis=(1,2)), a_min=1, a_max=None))/target_maps.shape[0]
            #fp_list[-1].append(fp)

            fn = np.sum(np.sum(fn_maps, axis=(1,2))/np.clip(np.sum(target_maps,axis=(1,2)), a_min=1, a_max=None))/target_maps.shape[0]
            #fn_list[-1].append(fn)

            precision = np.sum(np.sum(tp_maps,axis=(1,2))/(np.sum(peak_maps, axis=(1,2))+1e-8))/target_maps.shape[0]
            #precision_list.append(precision)
            ext_precision = np.sum(np.sum(intersection_maps,axis=(1,2))/(np.sum(extended_peak_maps, axis=(1,2))+1e-8))/target_maps.shape[0]
            #ext_precision_list.append(ext_precision)

            ext_recall = np.sum(np.sum(intersection_maps,axis=(1,2))/(np.sum(extended_target_maps, axis=(1,2))))/target_maps.shape[0]
            #ext_recall_list[-1].append(ext_recall)
            iou = np.sum(np.sum(intersection_maps,axis=(1,2))/np.sum(union_maps, axis=(1,2)))/target_maps.shape[0]
            #iou_list[-1].append(iou)
            niou = np.sum(np.sum(nintersection_maps,axis=(1,2))/np.clip(np.sum(extended_peak_maps, axis=(1,2)),a_min=1,a_max=None))/target_maps.shape[0]
            #niou_list[-1].append(niou)

            rows[m*len(factors)+n] = np.array([tags, data_offset, factor, 
                                                precision, recall, 
                                                 #ext_precision.get(), ext_recall.get(), 
                                                 fp, fn])
                                                 #iou.get(), niou.get()])#, kernel, detection_area]



    return rows#, peak_maps, detection_maps, fp_maps, intersection_maps, union_maps, nintersection_maps, cfar_maps

def cfar_detection_gpu(data, target_maps, kernel, factors, offsets, hard_threshold=None, detection_area=None, normalize=False, tags='Tag'):

    precision_list = []
    recall_list = []
    ext_precision_list = []
    ext_recall_list = []
    fp_list = []
    fn_list = []
    iou_list = []
    niou_list = []

    rows = []
    rows = np.zeros((len(factors)*len(offsets), 7))

    #data = cp.array(data)
    #target_maps = cp.array(target_maps)
    #detection_area = cp.array(detection_area)
    #kernel = cp.array(kernel)

    for m, data_offset in enumerate(offsets):
        
        #precision_list.append([])
        #recall_list.append([])
        #ext_precision_list.append([])
        #ext_recall_list.append([])
        #fp_list.append([])
        #fn_list.append([])
        #iou_list.append([])
        #niou_list.append([])


        proc_data = data + data_offset
        if normalize:
            proc_data = cp.einsum('ijk,i->ijk', proc_data, 1/np.max(data,axis=(1,2)))
    
        # CA - CFAR
        cfar_thresh_maps = cp.zeros_like(target_maps)
        for i in range(data.shape[0]):
            cfar_thresh_maps[i] = (cupyx.scipy.signal.convolve2d(cp.pad(proc_data[i], 
                                                               pad_width=((kernel.shape[0], kernel.shape[0]), (kernel.shape[1], kernel.shape[1])),
                                                                mode='symmetric'), 
                                                        kernel, mode='same')/cp.sum(kernel))[kernel.shape[0]:-kernel.shape[0],kernel.shape[1]:-kernel.shape[1]]

        if detection_area is not None:
            extended_target_maps = np.zeros_like(target_maps)
            for i in range(target_maps.shape[0]):
                extended_target_maps[i] = cp.clip(cupyx.scipy.signal.convolve2d(target_maps[i], detection_area, mode='same'), a_min=None, a_max=1)
        else:
            extended_target_maps = target_maps

        for n, factor in enumerate(factors):
            cfar_maps = cfar_thresh_maps*factor
            peak_maps = cp.array(data > (cfar_maps), dtype='float')

            #if detection_area is not None:
            #    extended_peak_maps = cp.zeros_like(peak_maps)
            #    for i in range(target_maps.shape[0]):
            #        extended_peak_maps[i] = cupyx.scipy.signal.convolve2d(peak_maps[i], detection_area, mode='same')>0
            #else:
            #    extended_peak_maps = peak_maps

            # Detection rate and false detection rate
            tp_maps = (peak_maps * target_maps) > 0
            fp_maps = (peak_maps - target_maps) > 0 # peak is there, but no target
            fn_maps = (target_maps - peak_maps) > 0 # target is there, but no peak

            #intersection_maps = (extended_peak_maps * extended_target_maps) > 0
            #union_maps = (extended_peak_maps + extended_target_maps) > 0
            #nintersection_maps = (extended_peak_maps - extended_target_maps) > 0

            recall = cp.sum(cp.sum(tp_maps,axis=(1,2))/cp.sum(target_maps, axis=(1,2)))/target_maps.shape[0]
            #recall_list[-1].append(recall)

            fp = cp.sum(cp.sum(fp_maps, axis=(1,2))/cp.clip(cp.sum(peak_maps,axis=(1,2)), a_min=1, a_max=None))/target_maps.shape[0]
            #fp_list[-1].append(fp)

            fn = cp.sum(cp.sum(fn_maps, axis=(1,2))/cp.clip(cp.sum(target_maps,axis=(1,2)), a_min=1, a_max=None))/target_maps.shape[0]
            #fn_list[-1].append(fn)

            precision = cp.sum(cp.sum(tp_maps,axis=(1,2))/(cp.sum(peak_maps, axis=(1,2))+1e-8))/target_maps.shape[0]
            #precision_list.append(precision)
            #ext_precision = cp.sum(cp.sum(intersection_maps,axis=(1,2))/(cp.sum(extended_peak_maps, axis=(1,2))+1e-8))/target_maps.shape[0]
            #ext_precision_list.append(ext_precision)

            #ext_recall = cp.sum(cp.sum(intersection_maps,axis=(1,2))/(cp.sum(extended_target_maps, axis=(1,2))))/target_maps.shape[0]
            #ext_recall_list[-1].append(ext_recall)
            #iou = cp.sum(cp.sum(intersection_maps,axis=(1,2))/cp.sum(union_maps, axis=(1,2)))/target_maps.shape[0]
            #iou_list[-1].append(iou)
            #niou = cp.sum(cp.sum(nintersection_maps,axis=(1,2))/cp.clip(cp.sum(extended_peak_maps, axis=(1,2)),a_min=1,a_max=None))/target_maps.shape[0]
            #niou_list[-1].append(niou)

            #rows.append([data_offset, factor, precision, recall, ext_precision, ext_recall, fp, fn, iou, niou])#, kernel, detection_area])
            #for t in reversed(tags):
            #    rows[-1].insert(0,t)
            rows[m*len(factors)+n] = np.array([tags, data_offset.get(), factor.get(), 
                                                precision.get(), recall.get(), 
                                                 #ext_precision.get(), ext_recall.get(), 
                                                 fp.get(), fn.get()])
                                                 #iou.get(), niou.get()])#, kernel, detection_area]

    return rows#, peak_maps, detection_maps, fp_maps, intersection_maps, union_maps, nintersection_maps, cfar_maps

def cfar_detection_pertarget_gpu(data, target_maps, kernel, factors, offsets, hard_threshold=None, detection_area=None, normalize=True, tags='Tag'):

    precision_list = []
    recall_list = []
    ext_precision_list = []
    ext_recall_list = []
    fp_list = []
    fn_list = []
    iou_list = []
    niou_list = []

    rows = []
    rows = np.zeros((len(factors)*len(offsets)*target_maps.shape[0], 8))

    target_maps = target_maps[...,:200,:]
    data = data[...,:200,:]
    #data = cp.array(data)
    #target_maps = cp.array(target_maps)
    #detection_area = cp.array(detection_area)
    #kernel = cp.array(kernel)
    if normalize:
        data = cp.einsum('ijk,i->ijk', data, 1/np.max(data,axis=(1,2)))

    # CA - CFAR
    cfar_thresh_maps = cp.zeros_like(target_maps[0])
    for i in range(data.shape[0]):
        cfar_thresh_maps[i] = (cupyx.scipy.signal.convolve2d(cp.pad(data[i], 
                                                           pad_width=((kernel.shape[0], kernel.shape[0]), (kernel.shape[1], kernel.shape[1])),
                                                            mode='symmetric'), 
                                                    kernel, mode='same')/cp.sum(kernel))[kernel.shape[0]:-kernel.shape[0],kernel.shape[1]:-kernel.shape[1]]

    for n, factor in enumerate(factors):
        skip = False
        for m, offset in enumerate(offsets):


            if not skip:
                cfar_maps = (cfar_thresh_maps+offset)*factor
                peak_maps = cp.array(data > (cfar_maps), dtype='float')

                if detection_area is not None:
                    extended_peak_maps = cp.zeros_like(peak_maps)
                    for i in range(peak_maps.shape[0]):
                        extended_peak_maps[i] = cupyx.scipy.signal.convolve2d(peak_maps[i], detection_area, mode='same')>0
                    extended_peak_maps = cp.repeat(cp.expand_dims(extended_peak_maps, axis=0), target_maps.shape[0], axis=0)

                    peak_maps = cp.repeat(cp.expand_dims(peak_maps, axis=0), target_maps.shape[0], axis=0)
                    tp_maps = (extended_peak_maps * target_maps) > 0 # peak is there and target is there
                else:
                    # Detection rate and false detection rate
                    peak_maps = cp.repeat(cp.expand_dims(peak_maps, axis=0), target_maps.shape[0], axis=0)
                    tp_maps = (peak_maps * target_maps) > 0 # peak is there and target is there

                fp_maps = (peak_maps - target_maps) > 0 # peak is there, but no target
                fn_maps = (target_maps - peak_maps) > 0 # target is there, but no peak


                recall = cp.sum(cp.sum(tp_maps,axis=(2,3))/(cp.sum(target_maps, axis=(2,3))+1e-8), axis=1)/target_maps.shape[1]

                fp = cp.sum(cp.sum(fp_maps, axis=(2,3))/cp.clip(cp.sum(peak_maps,axis=(2,3)), a_min=1, a_max=None), axis=1)/target_maps.shape[1]

                fn = cp.sum(cp.sum(fn_maps, axis=(2,3))/cp.clip(cp.sum(target_maps,axis=(2,3)), a_min=1, a_max=None), axis=1)/target_maps.shape[1]

                precision = cp.sum(cp.sum(tp_maps,axis=(2,3))/(cp.sum(peak_maps, axis=(2,3))+1e-8), axis=1)/target_maps.shape[1]

                for target_idx in range(target_maps.shape[0]):
                    rows[m*len(factors)*target_maps.shape[0]+n*target_maps.shape[0]+target_idx] = \
                                                    np.array([tags, target_idx, offset.get(), factor.get(), 
                                                    precision[target_idx].get(), recall[target_idx].get(), 
                                                     #ext_precision.get(), ext_recall.get(), 
                                                 fp[target_idx].get(), fn[target_idx].get()])
                                                 #iou.get(), niou.get()])#, kernel, detection_area]
                if np.sum(peak_maps) < 1:
                    skip = True
            else:
                for target_idx in range(target_maps.shape[0]):
                    rows[m*len(factors)*target_maps.shape[0]+n*target_maps.shape[0]+target_idx] = \
                                                    np.array([tags, target_idx, offset.get(), factor.get(), 
                                                    0, 0, 
                                                     #ext_precision.get(), ext_recall.get(), 
                                                    0, 1])

    return rows#, peak_maps, detection_maps, fp_maps, intersection_maps, union_maps, nintersection_maps, cfar_maps

def pixel_errors(target_maps, peak_maps):

    pixel_errors = []
    pixel_mean_errors = []
    pixel_std_errors = []
    pixel_min_errors = []
    pixel_max_errors = []
    for target_map, peak_map in zip(target_maps, peak_maps):
        pixel_errors.append([])
        pixel_mean_errors.append([])
        pixel_std_errors.append([])
        pixel_max_errors.append([])
        pixel_min_errors.append([])
        for target_x, target_y in zip(*np.where(target_map>0)):
            peak_xs, peak_ys = np.where(peak_map>0)
            if peak_xs.size and peak_ys.size:
                dx = target_x - peak_xs
                dy = target_y - peak_ys
                d_error = np.sqrt(dx**2 + dy**2)
                pixel_errors[-1].append(np.array(d_error))
                pixel_mean_errors[-1].append(np.mean(d_error))
                pixel_std_errors[-1].append(np.std(d_error))
                pixel_max_errors[-1].append(np.max(d_error))
                pixel_min_errors[-1].append(np.min(d_error))
            else:
                pixel_errors[-1].append(np.nan)
                pixel_mean_errors[-1].append(np.nan)
                pixel_std_errors[-1].append(np.nan)
                pixel_max_errors[-1].append(np.nan)
                pixel_min_errors[-1].append(np.nan)


    return pixel_mean_errors, pixel_std_errors, pixel_max_errors, pixel_min_errors, pixel_errors





#def snr(data, target_maps, window_size):
#
#    n_frames, n_ranges, n_angles = target_maps.shape
#    n_frames, n_ranges, n_angles = data.shape
#
#    peaks = detect_peak_in_target_window(data, target_maps, window_size)
#    snrs = []
#
#    for f in range(n_frames):
#
#        idx = data[f] < 0
#        data[f, idx] = 0
#        data[f] = data[f]/np.max(data[f])
#
#        signal = data[f, peaks[f][0], peaks[f][1]]
#
#        noise = np.sum(data[f])-signal
#        snr = signal**2 / noise**2
#
#        snrs.append(snr)
#
#    return snrs




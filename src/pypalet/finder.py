
import matplotlib.pyplot as plt
import matplotlib.patches as mp

import cv2 as cv
import numpy as np
from scipy import optimize

from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian
from skimage.measure import regionprops
from skimage.morphology import label, closing, square
from skimage.segmentation import clear_border
from skimage.transform import downscale_local_mean, rotate

import shapely
from shapely.geometry import Polygon


class Params:
    PARAMS_DEF = dict(
        board_blur=1.5,
        board_value='hue',
        board_n_groups=3,
        board_regions_min_fraction=.2,
        board_regions_max_fraction=.99,
        board_simplify_fraction=.1,
        board_min_rotation=5.,
        palet_zoom_border_fraction=1./5,
        palet_min_radius_fraction=1.25/100,
        palet_max_radius_fraction=20./100,
        fit_circle_max_cost=5.,
        circle_min_n_points=5
    )

    def __init__(self):
        self.params = self.PARAMS_DEF.copy()

    def set_params(self, **params):
        for p, v in params.items():
            self.params[p] = v

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value


class Storage:
    def __init__(self):
        self.storage = {}

    def store(self, key, value):
        self.storage[key] = value

    def get_stored(self, key):
        return self.storage[key]


class Finder(Storage):

    BOARD_VALUE_POS = ['hue', 'saturation', 'value']

    def __init__(self, img, params):
        super().__init__()
        self.img = img
        self.storage = {}
        self.params = params

    def make_low_res(self, img):
        """Make image max 1000px in size."""
        factor = int(max(img.shape) / 1000)
        self.store('downscale_factor', factor)
        if factor != 1:
            # log.info('Downscaling by factor %d', factor)
            down = downscale_local_mean(img, (factor, factor, 1))/255
            return down
        return img

    def run(self):
        self.locate_board(self.img)
        warp = self.transform_image(self.img)
        self.store('board', warp)
        self.locate_palets(warp)

    def locate_palets(self, board):
        # Get hue
        value = rgb2hsv(board)[..., 0]

        threshold = threshold_otsu(value)
        self.store('palet_hue_threshold', threshold)
        mask = value < threshold

        # Close holes
        closed = closing(mask, square(4))
        clear_border(closed)

        # Label groups
        label_image = label(closed, connectivity=1)
        borders = np.logical_xor(mask, closed)
        label_image[borders] = -1
        self.store('palets_labels', label_image)

        min_radius = self.params['palet_min_radius_fraction'] * board.shape[0]

        palets_regions = []
        for region in regionprops(label_image):
            # Remove small regions
            if region['area'] < np.pi*min_radius**2:
                continue
            palets_regions.append(
                PaletRegion(region, board, label_image, self.params))

        for p in palets_regions:
            p.find_center()

        self.store('palets', palets_regions)

    def locate_board(self, img):
        low = self.make_low_res(img)

        # Select value
        v = self.params['board_value']
        if v not in self.BOARD_VALUE_POS:
            raise KeyError("board_value not supported")
        i = self.BOARD_VALUE_POS.index(v)
        value = rgb2hsv(low)[..., i]

        # Blur it
        blur = gaussian(value, sigma=self.params['board_blur'])

        # Find n groups of values
        thresholds = threshold_multiotsu(
            blur, classes=self.params['board_n_groups'])
        groups = np.digitize(blur, bins=thresholds)
        self.store('board_groups', groups)

        # Close holes
        closed = closing(groups, square(4))
        clear_border(closed)

        # Label groups
        label_image = label(closed, connectivity=1)
        borders = np.logical_xor(groups, closed)
        label_image[borders] = -1
        self.store('board_labels', label_image)

        # Loop over labels to find largest (except background)
        largest_region = None
        for region in regionprops(label_image):
            min_frac = self.params['board_regions_min_fraction']
            if region['area'] < min_frac*label_image.size:
                continue
            max_frac = self.params['board_regions_max_fraction']
            if region['bbox_area'] > max_frac*label_image.size:
                continue

            if largest_region is None:
                largest_region = region
            else:
                if region['area'] > largest_region['area']:
                    largest_region = region

        # Create mask
        board_mask = label_image == largest_region['label']
        self.store('board_mask', board_mask)

        board_mask_ = to_cv(board_mask)

        # Find contours
        contours, hier = cv.findContours(board_mask_, cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)
        # Keep one with largest standard deviation
        contour = contours[select_contour_max_std(contours)]

        # Simplify countour
        simp_frac = self.params['board_simplify_fraction']
        espilon = simp_frac * cv.arcLength(contour, True)
        contour_simple = cv.approxPolyDP(contour, espilon, True)[:, 0, :]

        # Transform contour to large scale image
        contour_simple *= self.get_stored('downscale_factor')
        self.store('board_contour', contour_simple)

        if contour_simple.shape[0] != 4:
            raise IndexError("Contour not quadrilateral")

        # Rotate contour
        center = np.mean(contour_simple, 0)
        self.store('board_rotation_center', center)

        dif = contour_simple[1] - contour_simple[0]
        min_angle = self.params['board_min_rotation']
        angle = np.arctan(dif[1]/dif[0]) * 180/np.pi

        if (np.isclose(angle, 0., atol=min_angle) or
                np.isclose(angle, 90., atol=min_angle)):
            # log.info('No rotation')
            angle = 0.
        else:
            # log.info('Rotation by %f', angle)
            p = Polygon(contour_simple)
            p_rot = shapely.affinity.rotate(p, -angle, origin=tuple(center))
            contour_simple = np.array(p_rot.boundary.xy).T
            self.store('board_contour', contour_simple.T)
        self.store('board_rotation_angle', angle)

        # Find corners
        ul, ur, br, bl = [np.zeros(2) for _ in range(4)]
        for pt in contour_simple:
            top = pt[1] < center[1]
            left = pt[0] < center[0]
            if top and left:
                ul = pt
            elif top and ~left:
                ur = pt
            elif ~top and ~left:
                br = pt
            elif ~top and left:
                bl = pt
            else:
                raise ValueError("Point cannot be attributed to corner.")

        self.store('board_ul', ul)
        self.store('board_ur', ur)
        self.store('board_bl', bl)
        self.store('board_br', br)
        self.store('board_corners', np.float32([ul, ur, br, bl]))

    def transform_image(self, img):

        angle = self.get_stored('board_rotation_angle')
        if angle != 0:
            center = self.get_stored('board_rotation_center')
            rot = rotate(img, angle, center=center)
        else:
            rot = img

        ul = self.get_stored('board_ul')
        ur = self.get_stored('board_ur')
        bl = self.get_stored('board_bl')
        br = self.get_stored('board_br')

        # Get result size
        p = Polygon([ul, ur, br, bl])
        w = int(abs(p.bounds[2] - p.bounds[0]))
        h = int(abs(p.bounds[3] - p.bounds[1]))

        src = np.float32([ul, ur, bl, br])
        dst = np.float32(
            [[0, 0],
             [w, 0],
             [0, h],
             [w, h]]
        )

        mat = cv.getPerspectiveTransform(src, dst)
        res = cv.warpPerspective(rot, mat, (w, h))
        return res

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)

        ax1, ax2 = axes
        ax1.imshow(self.img)
        corners = self.get_stored('board_corners')
        ax1.scatter(corners[:, 0], corners[:, 1], color='r')

        ax2.imshow(self.get_stored('board'))
        for p in self.get_stored('palets'):
            x0, y0 = p.get_stored('x0'), p.get_stored('y0')
            for c in p.get_stored('contours'):
                ax2.add_artist(mp.Polygon(c[:, 0, :] + np.array([x0, y0]),
                                          fill=None, color='b', lw=.5))
            for c, R in p.get_params():
                ax2.scatter(*c, color='r')
                ax2.add_artist(plt.Circle(c, R, fill=None, color='r', lw=0.7))


class PaletRegion(Storage):
    def __init__(self, regionprops, img, label_image, params):
        super().__init__()
        self.params = params
        self.regionprops = regionprops
        self.get_region(img, label_image)
        self.store('img_shape', img.shape)

    def get_region(self, img, label_image):
        bbox = self.regionprops['bbox']
        minr, minc, maxr, maxc = bbox
        border = (maxr-minr) * self.params['palet_zoom_border_fraction']
        x0 = max(0, int(minc-border))
        x1 = min(img.shape[0], int(maxc+border))
        y0 = max(0, int(minr-border))
        y1 = min(img.shape[1], int(maxr+border))
        self.store('x0', x0)
        self.store('x1', x1)
        self.store('y0', y0)
        self.store('y1', y1)

        self.store('zoom', img[y0:y1, x0:x1].copy())
        self.store('label', label_image[y0:y1, x0:x1].copy())

    def find_center(self):
        zoom = self.get_stored('zoom')
        mask = self.get_stored('label') == self.regionprops['label']
        zoom[~mask] = 0

        # Remove shadows
        value = rgb2hsv(zoom)[..., 2]
        threshold = threshold_otsu(value[mask])
        top = value > threshold
        # Check we do not remove too much
        if np.sum(top) < 0.8 * np.sum(mask):
            # log.info("Top not found")
            threshold = 0
            top = value > 0
        self.store('threshold', threshold)

        top_ = to_cv(top)

        # Find contour
        contours, hier = cv.findContours(top_, cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)
        self.store('contours', contours)

        x0, y0 = self.get_stored('x0'), self.get_stored('y0')
        params = []
        for c in contours:
            if c.shape[0] < self.params['circle_min_n_points']:
                continue
            try:
                new_params = [self.fit_circle(c[:, 0, :])]
            except RuntimeError:
                try:
                    new_params = self.fit_2_circles(c[:, 0, :])
                except RuntimeError:
                    continue

            for (xc, yc), r in new_params:
                img_shape = self.get_stored('img_shape')
                w = (img_shape[0] + img_shape[1]) / 2.
                r_min = self.params['palet_min_radius_fraction'] * w
                r_max = self.params['palet_max_radius_fraction'] * w
                if not (r_min < r < r_max):
                    continue
                if (x0+xc < 0 - 3*r_max) or (x0+xc > img_shape[0] + 3*r_max):
                    continue
                if (y0+yc < 0 - 3*r_max) or (y0+yc > img_shape[1] + 3*r_max):
                    continue

                params.append(((xc, yc), r))

        self.store('params', params)

    def get_params(self):
        out = []
        for (xc, yc), r in self.get_stored('params'):
            xc += self.get_stored('x0')
            yc += self.get_stored('y0')
            out.append(((xc, yc), r))
        return out

    def fit_circle(self, points):
        x = points[:, 0]
        y = points[:, 1]

        def calc_R(xc, yc):
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def residuals(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center = np.mean(points, axis=0)
        res = optimize.least_squares(residuals, tuple(center))

        R = calc_R(*res.x).mean()
        goodness = np.sum(residuals(center)**2)/R**2

        if goodness > self.params['fit_circle_max_cost']:
            res.success = False

        if not res.success:
            raise RuntimeError("Could not fit circle.")

        return res.x, R

    def fit_2_circles(self, points):
        x = points[:, 0]
        y = points[:, 1]

        def calc_R(xc, yc):
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def calc_R_min(c1, c2):
            return np.fmin(calc_R(*c1), calc_R(*c2))

        def residuals(centers):
            R = calc_R_min(centers[:2], centers[2:])
            return R - R.mean()

        xc, yc = np.mean(points, axis=0)
        # We should expect major_axis/4 but 5 yields better results
        dist = self.regionprops['major_axis_length'] / 5
        angle = self.regionprops['orientation'] + np.pi/2.
        xc1 = xc + dist*np.cos(angle)
        yc1 = yc + dist*np.sin(angle)
        xc2 = xc - dist*np.cos(angle)
        yc2 = yc - dist*np.sin(angle)

        res = optimize.least_squares(residuals, (xc1, yc1, xc2, yc2))

        if not res.success:
            raise RuntimeError("Could not fit circles.")

        c1 = tuple(res.x[:2])
        c2 = tuple(res.x[2:])
        R = calc_R_min(c1, c2).mean()

        return [(c1, R), (c2, R)]

    def plot(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
        ax.imshow(self.get_stored('zoom'))
        for c in self.get_stored('contours'):
            p = mp.Polygon(self.get_stored('contours'), fill=None, ec='b')
            ax.add_artist(p)
        # ax.scatter(*self.center, color='r')


def select_contour_max_std(contours) -> int:
    std_m = 0.
    i_m = 0
    for i, c in enumerate(contours):
        std = np.std(np.linalg.norm(c, axis=2))
        if std > std_m:
            i_m = i
            std_m = std
    return i_m


def to_cv(array):
    # This is the way I found to convert a boolean mask to a cv 8bit matrix
    out = cv.cvtColor(array.astype('uint8')*255, cv.CV_8U)
    out = cv.cvtColor(out, cv.COLOR_RGB2GRAY)
    return out

from __future__ import division
from imageutils.transform import four_point_transform, box_image
from imageutils import imutils
from skimage.filters import threshold_adaptive
import argparse, cv2, sys, numpy

scale_factor = 500

class EdgeNotFound(Exception): pass
class ImageNotReadable(Exception): pass
class ImageNotFound(Exception): pass

cv2version = int(cv2.__version__.split(".")[0])
def _findContours(p1, p2, p3):
	if cv2version is 3:
		return cv2.findContours(p1, p2, p3)[1]
	elif cv2version is 2:
		return cv2.findContours(p1, p2, p3)[0]

def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required = True, help = "Path to the image to be scanned")
	ap.add_argument("-o", "--output", required = True, help = "Path to the image to be saved, or 'screen' for showing each step on screen")
	
	args = vars(ap.parse_args())

	work_image = cv2.imread(args["input"])
	if work_image is None: raise ImageNotFound()
	original_image = work_image.copy()
	output_image_name = args["output"]

	if output_image_name == 'screen': show = True
	else: show = False

	try:
		image_try = filter_image(work_image)
		contours = get_contours(image_try)
		# overlay(image, contours, show=True)
		cropped_image = crop_image(original_image, contours)
	except ImageNotReadable:
		image_try = contrast_image(work_image)
		contours = get_contours(image_try, polydb=0.03)	
		cropped_image = crop_image(original_image, contours)

	if not show: cv2.imwrite(output_image_name, cropped_image)

	# sys.exit('Success')
	sys.exit(1)

def wrapper(work_image, output_image_name):
	try:
		image_try = filter_image(work_image)
		contours = get_contours(image_try)
		cropped_image = crop_image(original_image, contours)
	except ImageNotReadable:
		image_try = contrast_image(work_image)
		contours = get_contours(image_try, polydb=0.03)	
		cropped_image = crop_image(original_image, contours)
	except:
	    return 0
	cv2.imwrite(output_image_name, cropped_image)	
	return cropped_image

def filter_image(image, canny1=5, canny2=5):
	# compute the ratio of the old height to the new height, and resize it
	image = imutils.resize(image, height=scale_factor, interpolation=cv2.INTER_NEAREST)
	# convert the image to grayscale, blur it, and find edges in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, canny1, canny2)
	return edged

def contrast_image(image, thresh1=180, thresh2=200, distance=50):
	def find_if_close(cnt1, cnt2):
		row1, row2 = cnt1.shape[0], cnt2.shape[0]
		for i in xrange(row1):
			for j in xrange(row2):
				dist = numpy.linalg.norm(cnt1[i]-cnt2[j])
				if abs(dist) < distance:
					return True
				elif i==row1-1 and j==row2-1:
					return False
	image = imutils.resize(image, height=scale_factor)
	# convert it to grayscale, and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.GaussianBlur(gray, (5, 5), 0)
	# threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
	thresh = cv2.threshold(gray2, thresh1, thresh2, cv2.THRESH_BINARY)[1]
	if distance:
		contours = _findContours(thresh, cv2.RETR_EXTERNAL, 2)
		LENGTH = len(contours)
		status = numpy.zeros((LENGTH,1))
		for i, cnt1 in enumerate(contours):
			x = i    
			if i != LENGTH-1:
				for j, cnt2 in enumerate(contours[i+1:]):
					x = x+1
					dist = find_if_close(cnt1, cnt2)
					if dist == True:
						val = min(status[i], status[x])
						status[x] = status[i] = val
					elif status[x]==status[i]:
						status[x] = i+1
		unified = []
		maximum = int(status.max())+1
		for i in xrange(maximum):
			pos = numpy.where(status==i)[0]
			if pos.size != 0:
				cont = numpy.vstack(contours[i] for i in pos)
				hull = cv2.convexHull(cont)
				unified.append(hull)
		cv2.drawContours(thresh, unified,-1,255,-1)
	return thresh

def get_contours(image, polydb=0.1, contour_range=7):
	# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
	contours = _findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:contour_range]
	# loop over the contours
	screenCnt = None
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True) #finds the Contour Perimeter
		approx = cv2.approxPolyDP(c, polydb * peri, True)
		# if our approximated contour has four points, then we can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
	if screenCnt is None:
		raise EdgeNotFound()
	# sometimes the algorythm finds a strange non-convex shape. The shape conforms to the card but its not complete, so then just complete the shape into a convex form
	if not cv2.isContourConvex(screenCnt):
		screenCnt = cv2.convexHull(screenCnt)
		x,y,w,h = cv2.boundingRect(screenCnt)
		screenCnt = numpy.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
	return screenCnt

def overlay(image, contours):
	new_image = imutils.resize(image, height=scale_factor)
	cv2.drawContours(new_image, [contours], -1, (2, 2, 251), 2)
	return new_image

def crop_image(image, contours, min_aspect_ratio=0.5):
	ratio = image.shape[0] / float(scale_factor)
	warped = four_point_transform(image, contours.reshape(4, 2) * ratio)
	# test to see if the box ratio is correct
	height, width, channels = warped.shape
	if height > width:
		aspect_ratio = width / height
	else:
		aspect_ratio = height / width
	if aspect_ratio < min_aspect_ratio:
		raise ImageNotReadable()
	# test to see if the orientation is correct, if not flip it
	if height > width:
		warped = cv2.transpose(warped)
		warped = cv2.flip(warped, 0)
	return warped

if __name__ == '__main__':
	try:
		main()
	except EdgeNotFound: sys.exit(0) #sys.exit('Edge not found')
	except ImageNotReadable: sys.exit(0) #sys.exit('Image not readable')
	except ImageNotFound: sys.exit(0) #sys.exit('Image not found')
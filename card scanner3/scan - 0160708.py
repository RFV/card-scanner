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
	if cv2version is 3: return cv2.findContours(p1, p2, p3)[1]
	elif cv2version is 2: return cv2.findContours(p1, p2, p3)[0]

def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required = True, help = "Path to the image to be scanned")
	ap.add_argument("-o", "--output", required = True, help = "Path to the image to be saved, or 'screen' for showing each step on screen")
	
	# ap.add_argument("-a", "--aspect_ratio", required = False, help = "Sets the min ratio. Requires aspect_ratio. 0.5")
	
	# ap.add_argument("-f", "--filter", required = False, help = "Return filtered image. Requires canny1,canny2. 10,10")
	# ap.add_argument("-c", "--contrast", required = False, help = "Return contrasted image. Requires thresh1,thresh2. 180,200")
	# ap.add_argument("-m", "--merge", required = False, help = "Return contrast merged image. Requires distance. 50")
	
	# ap.add_argument("-p", "--overlay", required = False, help = "Return border recognized image. Requires polydp. 0.03")
	# ap.add_argument("-P", "--cropped", required = False, help = "Returns the payload image. Requires polydp. 0.03" )

	args = vars(ap.parse_args())

	# aspect_ratio = args['aspect_ratio']
	# if not aspect_ratio: aspect_ratio = 0.5 #default

	work_image = cv2.imread(args["input"])
	if work_image is None: raise ImageNotFound()
	original_image = work_image.copy()
	output_image_name = args["output"]

	if output_image_name == 'screen': show = True
	else: show = False

	# if args["filter"]:
	# 	canny1, canny2 = args['filter'].split(',')
	# 	canny1 = int(canny1)
	# 	canny2 = int(canny2)
	# 	work_image, _ = filter_image(work_image, canny1=canny1, canny2=canny2, show=show)

	# elif args["contrast"]:
	# 	thresh1, thresh2 = args['contrast'].split(',')
	# 	thresh1 = int(thresh1)
	# 	thresh2 = int(thresh2)
		
	# 	if args["merge"]: distance = int(args['merge'])
	# 	else: distance = False

	# 	work_image = contrast_image(work_image, thresh1=thresh1, thresh2=thresh2, distance=distance, show=show)

	# if args["overlay"]:
	# 	polydb = float(args["overlay"])
	# 	work_image = overlay(original_image, get_contours(work_image, polydb=polydb), show=show)

	# elif args["cropped"]:
	# 	polydb = float(args["cropped"])
	# 	work_image = cropped_image(original_image, get_contours(work_image, polydb=polydb), show=show)

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

	sys.exit('Success')

def filter_image(image, canny1=5, canny2=5, show=False):
	# compute the ratio of the old height to the new height, and resize it
	image = imutils.resize(image, height=scale_factor, interpolation=cv2.INTER_NEAREST)

	# convert the image to grayscale, blur it, and find edges in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, canny1, canny2)

	# show the image(s)
	if show:
		cv2.imshow("Edged", edged)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return edged

def contrast_image(image, thresh1=180, thresh2=200, distance=50, show=False):
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
					else:
						if status[x]==status[i]:
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

	if show is True: #this is for debugging puposes
		cv2.imshow("Contrast", thresh)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return thresh

def get_contours(image, polydb=0.1, contour_range=7, show=False):
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

	if show: #this is for debugging puposes
		new_image = image.copy()
		cv2.drawContours(new_image, [screenCnt], -1, (255, 255, 0), 2)
		cv2.imshow("Contour1 image", new_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return screenCnt

def overlay(image, contours, show=False):
	new_image = imutils.resize(image, height=scale_factor)
	cv2.drawContours(new_image, [contours], -1, (2, 2, 251), 2)
	if show: #this is for debugging puposes
		cv2.imshow("Contour1 image", new_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return new_image

def crop_image(image, contours, min_aspect_ratio=0.5, show=False):
	ratio = image.shape[0] / float(scale_factor)
	warped = four_point_transform(image, contours.reshape(4, 2) * ratio)

	# test to see if the box ratio is correct
	height, width, channels = warped.shape
	if height > width: aspect_ratio = width / height
	else: aspect_ratio = height / width
	if aspect_ratio < min_aspect_ratio: raise ImageNotReadable()

	# test to see if the orientation is correct, if not flip it
	if height > width:
		warped = cv2.transpose(warped)
		warped = cv2.flip(warped, 0)

	if show: #this is for debugging puposes
		cv2.imshow("Payload", warped)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return warped

# def last_try(image, scale_factor):
# 	# Auto Canny
# 	def auto_canny(image, sigma=0.33):
# 		# compute the median of the single channel pixel intensities
# 		v = numpy.median(image)
	 
# 		# apply automatic Canny edge detection using the computed median
# 		lower = int(max(0, (1.0 - sigma) * v))
# 		upper = int(min(255, (1.0 + sigma) * v))
# 		print lower, upper
# 		edged = cv2.Canny(image, lower, upper)
	 
# 		# return the edged image
# 		return edged

# 	def get_screencnt(edged):
# 		# Find the contours in reverse order of area. Assuming the Card takes up majority of the picture
# 		cnts = _findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# 		# Go through and find the largest rectangle
# 		screenCnt = None

# 		for c in cnts:
# 			# Approximate the contour
# 			peri = cv2.arcLength(c, True)
# 			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		 
# 		 	# The card has four points, break if found
# 			if len(approx) == 4:
# 			 	screenCnt = approx
# 			 	break

# 	 	return screenCnt

# 	# Get the edges
# 	resized = imutils.resize(image, height=scale_factor, interpolation=cv2.INTER_NEAREST)
# 	grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# 	blurred = cv2.GaussianBlur(grayed, (5,5), 0)

# 	edged = auto_canny(blurred)

# 	screenCnt = get_screencnt(edged)
# 	if screenCnt == None:
# 		newEdged = cv2.Laplacian(blurred, cv2.CV_8U)
# 		screenCnt = get_screencnt(newEdged)

# 	return screenCnt

# def test(image_file):
# 		image = cv2.imread(image_file)
# 		try:
# 			image_try = filter_image(image)
# 			contours = get_contours(image_try)
# 			overlay(image, contours, show=True)
# 			cropped_image(image, contours, show=True)
# 		except ImageNotReadable:
# 			image_try = contrast_image(image)
# 			contours = get_contours(image_try, polydb=0.03)	
# 			overlay(image, contours, show=True)
# 			cropped_image(image, contours, show=True)

# test('1.jpg')
# test('5.jpg')
# test('6.jpg')
# test('7.jpg')
# test('8.jpg')
# test('9.jpg')
# test('D.jpg')
# test('E.jpg')

# for x in xrange(1,7):
# 	for y in xrange(1,7):
# 		for z in xrange(1,9):
# 			print x, y, z
# 			try:
# 				test(True, x*2, y*2, 0, z*0.02, 'thumbnail.jpg')
# 			except EdgeNotFound: pass

if __name__ == '__main__':
	try:
		main()
	except EdgeNotFound: sys.exit('Edge not found')
	except ImageNotReadable: sys.exit('Image not readable')
	except ImageNotFound: sys.exit('Image not found')
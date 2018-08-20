/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facerecognitiontest;

import com.sun.org.apache.xerces.internal.impl.dv.util.Base64;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32F;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_dnn;
import static org.bytedeco.javacpp.opencv_dnn.blobFromImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.core.CvType;

/**
 *
 * @author Innogeecks
 */
public class MatTestClass {

    private static final String PROTO_FILE = "resources/deploy.prototxt";
    private static final String CAFFE_MODEL_FILE = "resources/res10_300x300_ssd_iter_140000.caffemodel";
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static opencv_dnn.Net net = null;
    private static int count = 0;
    private static Mat imageMat;
    private static byte[] data;

    public static void main(String[] args) {
        try {
            BufferedImage bImage = ImageIO.read(new File("input.jpg"));
            int width = bImage.getWidth();
            int height = bImage.getHeight();
            System.out.println("Width: " + width + " Height: " + height);
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ImageIO.write(bImage, "jpeg", bos);
            String base64String = Base64.encode(bos.toByteArray());
            data = Base64.decode(base64String);
            imageMat = new Mat(new Size(width,height), CvType.CV_8UC4);
            imageMat.data().put(data);
//            ByteArrayInputStream bis = new ByteArrayInputStream(data);
//            BufferedImage bImage2 = ImageIO.read(bis);
//            ImageIO.write(bImage2, "jpg", new File("output.jpg"));
//            System.out.println("image created");
//            imageMat = new Mat ();
//            System.out.println(imageMat.toString());
//            detectAndDraw(imageMat);
            imwrite("output.png",imageMat);
            bos.flush();
            bos.close();

        } catch (IOException ex) {
            Logger.getLogger(MatTestClass.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static void detectAndDraw(Mat image) {//detect faces and draw a blue rectangle arroung each face

        resize(image, image, new opencv_core.Size(300, 300));//resize the image to match the input size of the model

        //create a 4-dimensional blob from image with NCHW (Number of images in the batch -for training only-, Channel, Height, Width) dimensions order,
        //for more detailes read the official docs at https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
        Mat blob = blobFromImage(image, 1.0, new opencv_core.Size(300, 300), new opencv_core.Scalar(104.0, 177.0, 123.0, 0), false, false);

        net.setInput(blob);//set the input to network model
        Mat output = net.forward();//feed forward the input to the netwrok to get the output matrix

        Mat ne = new Mat(new opencv_core.Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));//extract a 2d matrix for 4d output matrix with form of (number of detections x 7)

        FloatIndexer srcIndexer = ne.createIndexer(); // create indexer to access elements of the matric
        int faces = 0;
        for (int i = 0; i < output.size(3); i++) {
            float confidence = srcIndexer.get(i, 2);
            if (confidence > .6) {
                faces++;
            }
        }
        System.out.println("Faces no: " + faces);
        for (int i = 0; i < output.size(3); i++) {//iterate to extract elements

            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);

            if (faces > 1) {
                System.out.println("exit");
            }
            if (confidence > .6) {
                float tx = f1 * 300;//top left point's x
                float ty = f2 * 300;//top left point's y
                float bx = f3 * 300;//bottom right point's x
                float by = f4 * 300;//bottom right point's y
                Mat face = new Mat(image, new opencv_core.Rect(new opencv_core.Point((int) tx, (int) ty), new opencv_core.Point((int) bx, (int) by)));
                resize(face, face, new opencv_core.Size(300, 300));
                Mat out = new Mat();
                cvtColor(face, out, CV_BGR2GRAY);
//                imwrite("test/gray.png", out);
                rectangle(image, new opencv_core.Rect(new opencv_core.Point((int) tx, (int) ty), new opencv_core.Point((int) bx, (int) by)), new opencv_core.Scalar(255, 0, 0, 0));//print blue rectangle 
            }
        }
    }
}

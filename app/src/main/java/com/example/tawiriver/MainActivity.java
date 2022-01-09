package com.example.tawiriver;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    CameraBridgeViewBase javaCameraView;
    Mat mRGBA;
    Mat old_gray_frame = null;
    Boolean pointSelected = true;
    MatOfPoint2f old_points;
    MatOfPoint2f new_points;
    Point initial_point;
    MatOfByte status;
    MatOfFloat err;
    // Approximated value.
    double mpi = 0.05;
    long startTime;

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRGBA = inputFrame.rgba();

        if (this.old_gray_frame == null) {
            this.old_gray_frame = inputFrame.gray();
            ArrayList<Point> good_new = new ArrayList<>();
            good_new.add(new Point(255, 255));
            this.old_points.fromList(good_new);
        }

        if (this.pointSelected) {
            Imgproc.circle(
                    mRGBA, this.initial_point, 5,
                    new Scalar(255, 0, 0), 2);
            Video.calcOpticalFlowPyrLK(
                    this.old_gray_frame, inputFrame.gray(), this.old_points, this.new_points, this.status,
                    this.err, new Size(10, 10), 2, new TermCriteria(
                            TermCriteria.COUNT + TermCriteria.EPS, 10, 0.03));
            inputFrame.gray().copyTo(this.old_gray_frame);
            this.new_points.copyTo(this.old_points);
            Imgproc.circle(mRGBA, this.new_points.toArray()[0], 5, new Scalar(0, 0, 255), 2);
            updateSpeed(this.new_points);
        }

        return mRGBA;
    }

    private void updateSpeed(MatOfPoint2f newPosition) {
        // This calculations considers approximations.
        double x = newPosition.toList().get(0).x;
        double y = newPosition.toList().get(0).y;

        double distance = ((x - this.initial_point.x) * (x - this.initial_point.y)) +
                ((y - this.initial_point.y) * (y - this.initial_point.y));
        distance = Math.sqrt(distance);
        double distance_km = distance * 0.001 * this.mpi;
        double time_s = (System.currentTimeMillis() - this.startTime) * 0.001;
        double time_m = time_s / 60;
        double time_hr = time_m / 60;
        TextView speedView = findViewById(R.id.river_speed);
        speedView.setText(String.valueOf(Math.round(distance_km / time_hr)).concat(" km/hr"));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        javaCameraView = findViewById(R.id.cameraView);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCameraPermissionGranted();
        javaCameraView.setCvCameraViewListener(MainActivity.this);

        if (OpenCVLoader.initDebug()) {
            Log.d("TAG", "OpenCV Loaded");
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        } else {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseLoaderCallback);
        }

        this.old_points = new MatOfPoint2f();
        this.new_points = new MatOfPoint2f();
        this.status = new MatOfByte();
        this.err = new MatOfFloat();
        this.initial_point = new Point(255, 255);
        this.startTime = System.currentTimeMillis();
    }

    BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == BaseLoaderCallback.SUCCESS) {
                javaCameraView.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (OpenCVLoader.initDebug()) {
            Log.d("TAG", "OpenCV Loaded");
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        } else {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseLoaderCallback);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRGBA = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {

    }
}

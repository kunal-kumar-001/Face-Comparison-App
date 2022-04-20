package com.example.kunal;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

public class MainActivity extends AppCompatActivity {



    protected Interpreter tflite;

    private int imageSizeX;
    private int imageSizeY;

    private int REQUEST_PICK_IMAGE1 = 1000;
    private int REQUEST_PICK_IMAGE2 = 1001;

    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;

    public Bitmap oribitmap,testbitmap;
    public Bitmap cropped;
    Uri imageUri;

    ImageView oriImage,testImage;
    Button buverify;

    TextView result_text;

    float[][] ori_embedding = new float[1][128];
    float[][] test_embedding = new float[1][128];




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
        {
            if(checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {

                requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 0);
            }
        }

        initComponents();
    }

    private void initComponents() {

        oriImage = (ImageView) findViewById(R.id.image1);
        testImage = (ImageView) findViewById(R.id.image2);
        buverify = (Button) findViewById(R.id.verify);
        result_text = (TextView) findViewById(R.id.result);

        try {
            tflite = new Interpreter(loadmodelfile(this));
        }
        catch(Exception e){
            e.printStackTrace();
        }

        oriImage.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");

            startActivityForResult(intent,REQUEST_PICK_IMAGE1);
        });

        testImage.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");

            startActivityForResult(intent,REQUEST_PICK_IMAGE2);
        });

        buverify.setOnClickListener(v -> {
            double distance = calculate_distance(ori_embedding,test_embedding);

            Toast.makeText(MainActivity.this,"Distance is " + distance,Toast.LENGTH_LONG).show();
            if(distance<9.0)
            {
                result_text.setText("Result : Same faces");
            }
            else
            {
                result_text.setText("Result : Different faces");
            }

        });
    }

    private double calculate_distance(float[][] ori_embedding,float[][] test_embedding)
    {
        double sum = 0.0;
        for(int i=0;i<128;i++)
        {
            sum = sum+Math.pow((ori_embedding[0][i] - test_embedding[0][i]),2.0);
        }
        return Math.sqrt(sum);
    }

    private TensorImage loadImage(Bitmap bitmap,TensorImage inputImageBuffer)
    {
        inputImageBuffer.load(bitmap);

        int cropSize = Math.min(bitmap.getWidth(),bitmap.getHeight());

        ImageProcessor imageProcessor = new ImageProcessor.Builder()

                                        .add(new ResizeOp(imageSizeX,imageSizeY,ResizeOp.ResizeMethod.BILINEAR))
                                        .add(new NormalizeOp(127.5f,127.5f))
                                        .build();

        return imageProcessor.process(inputImageBuffer);

    }

    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("facenet.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode == RESULT_OK)
        {
            if(requestCode == REQUEST_PICK_IMAGE1){
                Uri uri = data.getData();
                Bitmap bitmap = loadFromUri(uri);

                oriImage.setImageBitmap(bitmap);
                face_detector(bitmap,"original");
            }
            else if(requestCode == REQUEST_PICK_IMAGE2)
            {
                Uri uri = data.getData();
                Bitmap bitmap = loadFromUri(uri);

                testImage.setImageBitmap(bitmap);
                face_detector(bitmap,"test");
            }
        }

    }

    public void face_detector(Bitmap bitmap, String imageType) {
        Bitmap outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888,true);
        InputImage inputImage = InputImage.fromBitmap(outputBitmap,0);

        FaceDetectorOptions highAccuracyOpts =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .enableTracking()
                        .build();

        FaceDetector faceDetector = FaceDetection.getClient(highAccuracyOpts);

        faceDetector.process(inputImage).addOnSuccessListener(new OnSuccessListener<List<Face>>(){

            @Override
            public void onSuccess(List<Face> faces) {

                if(faces.isEmpty()) {
                 //   Log.d("check","No faces buddy!");
                }
                else
                {
                    for(Face face:faces) {
                        Rect bounds = face.getBoundingBox();
                        cropped = Bitmap.createBitmap(outputBitmap,bounds.left,bounds.top,bounds.width(),bounds.height());
                        get_embeddings(cropped,imageType);
                    }
                }
            }
        })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        e.printStackTrace();
                    }
                });


    }

    public void get_embeddings(Bitmap bitmap,String imagetype)
    {

        TensorImage inputImageBuffer;

        float[][] embedding = new float[1][128];

        int imageTensorIndex = 0;

        // {1,height,width,3}
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();

        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

        inputImageBuffer = new TensorImage(imageDataType);

        inputImageBuffer = loadImage(bitmap,inputImageBuffer);

        tflite.run(inputImageBuffer.getBuffer(),embedding);

        if(imagetype.equals("original"))
            ori_embedding = embedding;
        else if(imagetype.equals("test"))
            test_embedding = embedding;


    }

    private Bitmap loadFromUri(Uri uri)
    {
        Bitmap bitmap = null;

        try {
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.O_MR1) {
                ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(),uri);
                bitmap = ImageDecoder.decodeBitmap(source);
            }
            else
            {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(),uri);
            }
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

        return bitmap;
    }

}
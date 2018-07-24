package nl.ordina.jtech.hackadrone.io;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import com.sun.tools.doclets.internal.toolkit.util.DocFinder;
import nl.ordina.jtech.hackadrone.io.recognition.Prediction;



public class VideoFrame {

    private static final int FRAME_WIDTH = 720;
    private static final int FRAME_HEIGHT = 676;

    private JFrame frame;
    private JLabel label;
    private BufferedImage bufferedImage;

    private Socket socket;
    private OutputStream output;

    public VideoFrame() {
        frame = new JFrame("Video Frame");
        frame.setSize(FRAME_WIDTH, FRAME_HEIGHT);
        label = new JLabel();
        frame.add(label);

        try {
            socket = new Socket("localhost", 9090);
            socket.setKeepAlive(true);
            output = socket.getOutputStream();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void showFrame() {
        frame.setVisible(true);
    }

    public void hideFrame() {
        frame.setVisible(false);
    }

    public void updateVideoFrame(BufferedImage bufferedImage) {
        setBufferedImage(bufferedImage);
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        try {
            ImageIO.write(bufferedImage, "jpg", outputStream);
            byte[] size = ByteBuffer.allocate(4).putInt(outputStream.size()).array();
            output.write(size);
            output.write(outputStream.toByteArray());
            output.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        label.setIcon(new ImageIcon(bufferedImage));
    }

    public void setFrameLabel(List<Prediction> predictionList) {
        label.setIcon(new ImageIcon(bufferedImage));
        label.setVerticalTextPosition(JLabel.BOTTOM);
        label.setHorizontalTextPosition(JLabel.CENTER);
        label.setText(predictionList.toString());
    }

    public void setBufferedImage(BufferedImage bufferedImage) {
        this.bufferedImage = bufferedImage;
    }

}

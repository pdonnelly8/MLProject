import * as React from "react";
import { Text, View, StyleSheet, Button, Pressable } from "react-native";
import { Audio } from "expo-av";
import getPath from '@flyerhq/react-native-android-uri-path'

export default function App() {
  const [recording, setRecording] = React.useState();
  const [title] = React.useState("Stroke Detection App");
  const [text, setText] = React.useState("");

  async function startRecording() {
    try {
      setText("");
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const { recording } = await Audio.Recording.createAsync({
        android: {
          extension: '.m4a',
          outputFormat: Audio.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_MPEG_4,
          audioEncoder: Audio.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_AAC,
        },
        ios: {
          extension: ".wav",
          sampleRate: 44100,
          numberOfChannels: 2,
          bitRate: 128000,
          audioQuality: Audio.RECORDING_OPTION_IOS_AUDIO_QUALITY_HIGH,
          outputFormat: Audio.RECORDING_OPTION_IOS_OUTPUT_FORMAT_LINEARPCM,
        },
      });
      setRecording(recording);
    } catch (err) {
      console.error("Failed to start recording", err);
    }
  }

  async function stopRecording() {
    setText("Awaiting Result...");
    setRecording(undefined);
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();

    try {
      console.log("Uploading " + uri);
      let apiUrl = 'http://192.168.0.32:5000/audio';
      let uriParts = uri.split('.');
      let fileType = uriParts[uriParts.length - 1];

      let formData = new FormData();
      formData.append('file', {
        uri,
        name: `recording.${fileType}`,
        type: `audio/x-${fileType}`,
      });

      let options = {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'multipart/form-data',
        },
      };

      console.log("POSTing " + uri + " to " + apiUrl);
      
      const path = getPath(recording.getURI());
      console.log(path);
      
      //Send fetch request
      const response = await fetch(apiUrl, options);
      //setText to the body of response from fetch
      const body = await response.json();
      setText(body.text);
    } catch (err) {
      console.error(err);
    }  
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      <Pressable
        style={styles.button}
        onPress={recording ? stopRecording : startRecording}>
          <Text style={styles.buttonText}>{recording ? "Stop Recording" : "Start Recording"}</Text>
      </Pressable>
      <Text style={styles.text}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 35,
    fontWeight: "bold",
    alignContent: "stretch",
  },
  button: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 4,
    elevation: 3,
    backgroundColor: 'black',
  },
  buttonText: {
    fontSize: 16,
    lineHeight: 21,
    fontWeight: 'bold',
    letterSpacing: 0.25,
    color: 'white',
  },
  text: {
    fontSize: 16,
    lineHeight: 21,
    fontWeight: 'bold',
    letterSpacing: 0.25,
    color: 'black',
  },
});
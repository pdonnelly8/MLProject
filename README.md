# Pauric Donnelly Final Year Project - Detecting impaired speech to help identify stroke

## Introduction 
Stroke is the second largest cause of death and disability-adjusted life-years in the world. Impaired speech is a core symptom that medical professionals consider when identifying cases of stroke. Unfortunately, 1/3 of patients have a significant delay in treatment in part due to a failure of patients, first-responders and paramedics to recognise signs of stroke. A technology that can accurately detect speech impairment could support the diagnostic capabilities of non-experts and reduce delays in stroke treatment, since delays in seeking treatment for stroke is the major factor limiting the delivery of treatment. This project will aim to develop computational models capable of classifying the presence (or absence) of impairment in audio files. This project will also include collaborating with a Software Engineering project with a similar product to that which will be created and to incorporate these models with theirs. 

## Dataset
The dataset I will be using is the Saarbruecken Voice Database. This is a database with recordings of over 2000 people with different styles of audio recordings:

- Recording of the vowels [i, a, u] produced at normal, high and low pitch
- Recordings of the vowels [i, a, u] with rising-falling pitch
- Recording of the sentence ''Guten Morgen, wie geht es Ihnen?'' ("Good morning, how are you?")

This dataset is appropriate for this use case as it holds samples of both healthy audio and pathological audio, which will be able to be used within the various models to train, validate, and test each model. The Saarbruecken database is also a publicly available database, so procurement of the data was a simple procedure.

 A drawback of the Saarbruecken database is that a proportion of the pathological samples have very similar characteristics to that of the healthy audio, in that it was very hard to tell from listening to the speech sample that the subject had a speech disorder. It therefore became necessary to gather pathological audio samples that could be discernible to the ear as “unhealthy”. The sample size of full .wav audio files was shortened to a total of 202, with an even split of healthy and pathological data amongst them. 
 However, because of the nature of the audio samples, it became possible to apply a technique known as Data Augmentation, defined as “a suite of techniques that enhance the size and quality of training datasets such that Deep Learning models can be built using them”. Since audio recordings in the Saarbruecken database had 4 different pitches of vowel sounds per audio sample, it became possible to extract all different variations of vowel sounds so that for each full audio sample given, 4 audio samples could be generated. Through editing these audio files a total of 808 samples of audio were gathered. The directory hierarchy of these audio samples is saved as “AudioSamples/'AudioClass'/'PatientID'/'PatientID'\_'AudioClass'\_'SamplePitch'.wav” 

 ## Traversing the Repository

 For each aspect of the project there is their own directory. There are also READMEs within each directory explaining how to run each aspect.

 - To run the machine learning scripts and to see the results of these experiments, go to 'Machine_Learning'
 - To run the Front-End client and the testing measures for it, go to 'stroke-app'
 - To run the Back-End server that interacts with the Front-End client, go to 'flaskServer'
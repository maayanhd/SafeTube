import React, { useState, useEffect } from 'react';
import './App.css';
import logo from './images/logo.jpeg';

//components
import Form from "./components/Form";
import PlayerAndPlaylist from "./components/PlayerAndPlaylist";
import Description from './components/Description';
import SuggestedVideo from './components/SuggestedVideo';
import VideoNameAndRating from './components/VideoNameAndRating';
import VideoViewsAndLikes from './components/VideoViewsAndLikes';

const getIDFromURL = (videoURL) => {
  let startIndex = videoURL.indexOf("embed/") + 6;
  let endIndex = videoURL.length;
  if (videoURL.includes("&")) {
    endIndex = videoURL.indexOf("&");
  }
  return (videoURL.substring(startIndex, endIndex));
}

function App() {
  //states
  const [inputText, setInputText] = useState(""); // user playlist url
  const [videoPlaylist, setVideoPlaylist] = useState([]); // array that will hold videos that show up on the playlist
  const [firstVideoURL, setFirstVideoURL] = useState(""); // string of first video URL
  const [currentVideoIndexInPlaylist, setCurrentVideoIndexInPlaylist] = useState(0); // index of video in playlist
  const [currentVideoId, setCurrentVideoId] = useState(1); // video ID
  const [playBoxVisual, setPlayBoxVisual] = useState("hiddenPlayerBox"); // playbox hidden or seen
  const [suggestedVideosVisual, setSuggestedVideosVisual] = useState("hiddenPlaylistTable"); // suggested videos hidden or seen
  const [videoDescriptionVisual, setVideoDescriptionVisual] = useState("hiddenVideoDescriptionTextArea"); // video description hidden or seen
  const [currVideoDescription, setCurrVideoDescription] = useState(""); // Video Description
  const [suggestedPlaylistVideos, setSuggestedPlaylistVideos] = useState([<SuggestedVideo
    imgURL="https://i.ytimg.com/vi_webp/Mos5QJAje28/maxresdefault.webp" VideoID="Mos5QJAje28"
    setCurrentVideoId={setCurrentVideoId} videoName="dummyVideo" />]);
  const [channelDetailsDisplay, setChannelDetailsDisplay] = useState("channelDetailsHidden");
  const [videoDetailsDisplay, setvideoDetailsDisplay] = useState("videoDetailsHidden");
  const [loading,setLoading] = useState(false);



  return (
    <div className="App">
      <header>
        <img src={logo} width="70"></img>
        <h1>Safe Tube</h1>
      </header>

      <Form
        setCurrentVideoIndexInPlaylist={setCurrentVideoIndexInPlaylist}
        getIDFromURL={getIDFromURL} setVideoPlaylist={setVideoPlaylist}
        setPlayBoxVisual={setPlayBoxVisual} firstVideoURL={firstVideoURL} setFirstVideoURL={setFirstVideoURL}
        setInputText={setInputText} inputText={inputText} setSuggestedVideosVisual={setSuggestedVideosVisual}
        setChannelDetailsDisplay={setChannelDetailsDisplay} setvideoDetailsDisplay={setvideoDetailsDisplay}
        loading={loading} setLoading={setLoading}
      />

      <br></br><br></br>

      <PlayerAndPlaylist
        currentVideoIndexInPlaylist={currentVideoIndexInPlaylist} setCurrentVideoIndexInPlaylist={setCurrentVideoIndexInPlaylist}
        videoPlaylist={videoPlaylist} playBoxVisual={playBoxVisual} setPlayBoxVisual={setPlayBoxVisual}
        firstVideoURL={firstVideoURL} setFirstVideoURL={setFirstVideoURL} getIDFromURL={getIDFromURL}
        currentVideoId={currentVideoId} setCurrentVideoId={setCurrentVideoId} suggestedPlaylistVideos={suggestedPlaylistVideos}
        setSuggestedPlaylistVideos={setSuggestedPlaylistVideos} suggestedVideosVisual={suggestedVideosVisual}
        channelDetailsDisplay={channelDetailsDisplay} setChannelDetailsDisplay={setChannelDetailsDisplay}
      />
      <div className="centerDetails">
        <VideoNameAndRating
          videoDetailsDisplay={videoDetailsDisplay} videoPlaylist={videoPlaylist} currentVideoIndexInPlaylist={currentVideoIndexInPlaylist}
        />
      </div>
      <div className="centerDetails">
        <VideoViewsAndLikes
          videoDetailsDisplay={videoDetailsDisplay} views="13,232,132" likes={98} dislikes={49}
        />
      </div>
    </div>
  );
}
/*      <Description 
      getIDFromURL={getIDFromURL} videoDescriptionVisual={videoDescriptionVisual} 
      setVideoDescriptionVisual = {setVideoDescriptionVisual} firstVideoURL={firstVideoURL}
       currVideoDescription={currVideoDescription} setCurrVideoDescription={setCurrVideoDescription}
       currentVideoId={currentVideoId}
       />*/
export default App;

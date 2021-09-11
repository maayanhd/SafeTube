import React, { useEffect, useState } from 'react';
import CanvasJSReact from './canvasjs.react';

var CanvasJS = CanvasJSReact.CanvasJS;
var CanvasJSChart = CanvasJSReact.CanvasJSChart;

const ChannelDetails = ({ videoPlaylist, currentVideoIndexInPlaylist, channelDetailsDisplay }) => {


    //states
    const [badVidNum, setBadVidNum] = useState(0);
    const [goodVidNum, setGoodVidNum] = useState(1);
    const [chPop, setChPop] = useState(2);
    const [channelNameVar, setChannelNameVar] = useState("");

    let options = {
        title: {
            text: "Channel details"
        },
        width: 200,
        height: 400,
        data: [
            {

                type: "column",
                dataPoints: [
                    { label: "Safe videos", y: goodVidNum },

                    { label: "bad videos", y: badVidNum}
                    
                    
                ]
            }
        ]
    }

    useEffect(()=>{
        if(videoPlaylist[0]!=undefined){
            setBadVidNum(videoPlaylist[currentVideoIndexInPlaylist].chBadVidAmount);
            setGoodVidNum(videoPlaylist[currentVideoIndexInPlaylist].chGoodVidAmount);
            setChPop(0);
        }
    },[videoPlaylist])

    useEffect(() => {
        if (videoPlaylist.length > currentVideoIndexInPlaylist) {

            setChannelNameVar(videoPlaylist[currentVideoIndexInPlaylist].channelName);
        }

    }, [videoPlaylist, currentVideoIndexInPlaylist]);
    return (
        <div className={channelDetailsDisplay}>
            <div>
                <label className="dataLabel">Channel name: <b>{channelNameVar}</b> </label><br></br>
            </div>
            <div>
                <CanvasJSChart options={options}/>
            </div>
        </div>
    );
}

export default ChannelDetails;
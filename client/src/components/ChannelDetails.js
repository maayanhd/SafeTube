import React, { useEffect, useState } from 'react';
import CanvasJSReact from './canvasjs.react';

var CanvasJS = CanvasJSReact.CanvasJS;
var CanvasJSChart = CanvasJSReact.CanvasJSChart;

const ChannelDetails = ({ videoPlaylist, currentVideoIndexInPlaylist, channelDetailsDisplay }) => {


    //states
    const [channelNameVar, setChannelNameVar] = useState("chName");
    const [channelViewsVar, setChannelViewsVar] = useState("3,000,000,000");
    const [channelratingVar, setChannelratingVar] = useState("9.3");


    const options = {
        title: {
            text: "Basic Column Chart"
        },
        width: 200,
        height: 400,
        data: [
            {
                // Change type to "doughnut", "line", "splineArea", etc.
                type: "column",
                dataPoints: [
                    { label: "UnSafe videoss", y: 10 },
                    { label: "Sa ", y: 20 },
                    { label: "Safe videos", y: 25 },
                    { label: "Popularity", y: 40 }
                ]
            }
        ]
    }


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
                <CanvasJSChart options={options} />
            </div>
        </div>
    );
}

export default ChannelDetails;
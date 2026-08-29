/* Minimal Streamlit custom-component bridge (vanilla JS). */
(function (root) {
  const Streamlit = {
    setComponentReady: function () {
      window.parent.postMessage(
        { isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 },
        "*"
      );
    },
    setFrameHeight: function (height) {
      window.parent.postMessage(
        { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: height },
        "*"
      );
    },
    setComponentValue: function (value) {
      window.parent.postMessage(
        { isStreamlitMessage: true, type: "streamlit:setComponentValue", value: value },
        "*"
      );
    },
  };

  root.Streamlit = Streamlit;
})(window);

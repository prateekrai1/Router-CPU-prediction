package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/google/gopacket"
	"github.com/google/gopacket/layers"
	"github.com/google/gopacket/pcap"
)

// Log file path
const logFilePath = "replay_log.txt"

// Function to replay a single pcap file
func replayPcap(filePath string, handle *pcap.Handle) error {
	startTime := time.Now()

	// Open the pcap file
	pcapHandle, err := pcap.OpenOffline(filePath)
	if err != nil {
		return fmt.Errorf("error opening pcap file: %v", err)
	}
	defer pcapHandle.Close()

	packetSource := gopacket.NewPacketSource(pcapHandle, pcapHandle.LinkType())

	var totalBytes int64
	var tcpCount, udpCount int

	// Replay packets at 10x speed
	var lastTimestamp time.Time
	for packet := range packetSource.Packets() {
		// Extract the timestamp
		packetTime := packet.Metadata().Timestamp

		// If this is not the first packet, calculate sleep time
		if !lastTimestamp.IsZero() {
			delay := packetTime.Sub(lastTimestamp) / 10
			time.Sleep(delay)
		}
		lastTimestamp = packetTime

		// Count bytes for bandwidth calculation
		totalBytes += int64(len(packet.Data()))

		// Count TCP and UDP packets
		if layer := packet.Layer(layers.LayerTypeTCP); layer != nil {
			tcpCount++
		} else if layer := packet.Layer(layers.LayerTypeUDP); layer != nil {
			udpCount++
		}

		// Write the packet to the network interface
		err := handle.WritePacketData(packet.Data())
		if err != nil {
			log.Printf("Error sending packet: %v", err)
		}
	}

	// Calculate bandwidth
	duration := time.Since(startTime).Seconds()
	bandwidth := (float64(totalBytes) * 8 / duration) / 10 // Bits per second, divided by 10

	// Log replay details
	endTime := time.Now()
	logMessage := fmt.Sprintf(
		"File: %s | Start: %s | End: %s | Bandwidth/10: %.2f bps | TCP: %d | UDP: %d\n",
		filePath, startTime.Format(time.RFC3339), endTime.Format(time.RFC3339), bandwidth, tcpCount, udpCount,
	)
	logToFile(logMessage)

	fmt.Print(logMessage)
	return nil
}

// Function to log messages to a file
func logToFile(message string) {
	f, err := os.OpenFile(logFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Error opening log file: %v", err)
	}
	defer f.Close()

	if _, err := f.WriteString(message); err != nil {
		log.Fatalf("Error writing to log file: %v", err)
	}
}

// Function to replay all pcap files in a directory
func replayAllPcaps(inputDir string) {
	files, err := filepath.Glob(filepath.Join(inputDir, "*.pcap"))
	if err != nil {
		log.Fatalf("Error scanning directory: %v", err)
	}

	// Open a live network interface to send packets
	handle, err := pcap.OpenLive("wlp0s20f3", 65536, true, pcap.BlockForever)
	if err != nil {
		log.Fatalf("Error opening network interface: %v", err)
	}
	defer handle.Close()

	for _, file := range files {
		err := replayPcap(file, handle)
		if err != nil {
			log.Printf("Skipping file %s due to error: %v\n", file, err)
		}
	}
}

func main() {
	inputDir := "/home/umd-user/Downloads/Scripts/benign1" // Change to your pcap directory

	replayAllPcaps(inputDir)
}


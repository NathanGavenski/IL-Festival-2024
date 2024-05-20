# Turing Test for Imitation Learning (Festival of AI 2024)

## Requirements

They are all listed under `requiremenents.txt`

## How to run

Change the `HOST` under the `Client` class to connect to the IP if executing in two different machines.
Execute the server first, and after the client.

On one machine:
```{bash}
python server.py
```

On a second machine:
```{bash}
python client.py
```

## TODO

- [x] Create a Client Server relation
- [x] Make Client and Server render the same game
- [X] Buy a controller for the player
- [x] Listen to inputs to make the player act in the environment
- [x] Create a chance to the client not see the server but an agent replay
- [X] Find agents that look like humans
- [X] Record these agents
- [X] Create a survey for the viewer to respond after guessing
- [X] Make an interface for player
- [X] Make an interface for viewer

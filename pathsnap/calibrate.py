"""Recalibrate decoder from saved training-trip out-of-fold predictions only."""
from .train import recalibrate

def main():
    recalibrate()

if __name__=='__main__': main()

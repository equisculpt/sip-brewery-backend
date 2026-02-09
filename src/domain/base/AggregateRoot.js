/**
 * 🏛️ DOMAIN-DRIVEN DESIGN: AGGREGATE ROOT BASE CLASS
 * 
 * Base class for all domain aggregates implementing DDD patterns
 * Handles domain events and ensures invariant consistency
 * 
 * @author FSI Architecture Team
 * @version 3.0.0
 */

class AggregateRoot {
  constructor(id) {
    if (!id) {
      throw new Error('AggregateRoot requires an identifier');
    }

    this.id = id;
    this._domainEvents = [];
    this.version = 0;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }

  /**
   * Add domain event to be published
   */
  addDomainEvent(domainEvent) {
    if (!domainEvent) {
      throw new Error('Domain event cannot be null');
    }

    this._domainEvents.push(domainEvent);
  }

  /**
   * Retrieve and clear all domain events
   */
  pullDomainEvents() {
    const events = [...this._domainEvents];
    this._domainEvents = [];
    return events;
  }

  /**
   * Clear domain events without returning them
   */
  clearDomainEvents() {
    this._domainEvents = [];
  }

  /**
   * Get current domain events without clearing
   */
  getDomainEvents() {
    return [...this._domainEvents];
  }

  /**
   * Mark aggregate as modified
   */
  markModified() {
    this.updatedAt = new Date();
    this.version++;
  }

  /**
   * Check if aggregate has uncommitted events
   */
  hasUncommittedEvents() {
    return this._domainEvents.length > 0;
  }

  publishDomainEvents(dispatcher) {
    if (!dispatcher) return;
    dispatcher.dispatchEvents(this);
  }
}

module.exports = { AggregateRoot };
